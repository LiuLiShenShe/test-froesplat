#!/usr/bin/env python3
"""Import an actual viewing PLY into a local official SuperSplat build on CPU."""
import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from urllib.parse import urlsplit

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ForeSplat_AutoGaussianCleaner.cleaner.io import sha, write_json
from playwright.sync_api import sync_playwright


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


def vertex_count(path):
    with path.open('rb') as stream:
        while True:
            line = stream.readline()
            if not line:
                raise ValueError('Invalid PLY header')
            if line.startswith(b'element vertex '):
                return int(line.split()[-1])
            if line.strip() == b'end_header':
                raise ValueError('Missing PLY vertex count')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--editor-source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ply', type=Path)
    parser.add_argument('--timeout-seconds', type=int, default=180)
    args = parser.parse_args()
    source = args.editor_source.resolve()
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True)
    if not (source/'dist/index.html').exists():
        raise ValueError('Build the official local editor first')
    result = dict(status='browser_probe_running', source=str(source),
        editor_version=json.loads((source/'package.json').read_text())['version'],
        editor_commit=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip(),
        lockfile_sha256=sha(source/'package-lock.json'), script_sha256=sha(Path(__file__)),
        profile='Isolated temporary Chromium profile; no existing user browser/profile or shared CDP service used',
        network_policy='Only loopback requests to the local static editor are permitted; external requests are blocked, including model upload',
        started=time.time(), console=[], page_errors=[], blocked_requests=[])
    if args.ply:
        ply=args.ply.resolve();result.update(ply=str(ply),ply_sha256=sha(ply),expected_gaussians=vertex_count(ply))
    write_json(output/'BROWSER_AUDIT.json',result)
    server=ThreadingHTTPServer(('127.0.0.1',0),partial(QuietHandler,directory=str(source/'dist')))
    threading.Thread(target=server.serve_forever,daemon=True).start()
    port=server.server_port
    try:
        with sync_playwright() as playwright:
            browser=playwright.chromium.launch(executable_path='/usr/bin/google-chrome',headless=True,
                args=['--use-angle=swiftshader','--use-vulkan=swiftshader','--enable-unsafe-webgpu',
                      '--enable-unsafe-swiftshader','--enable-features=Vulkan','--disable-accelerated-video-decode'])
            context=browser.new_context(viewport={'width':1280,'height':900},device_scale_factor=1,service_workers='block')
            def route(request_route):
                request=request_route.request;address=urlsplit(request.url)
                if address.hostname=='127.0.0.1' and address.port==port and request.method in ['GET','HEAD']:
                    request_route.continue_()
                else:
                    result['blocked_requests'].append(dict(method=request.method,url=request.url))
                    request_route.abort()
            context.route('**/*',route)
            context.add_init_script("""(() => {
                Object.defineProperty(window, 'showOpenFilePicker', {value:undefined, configurable:true});
                window.audit_adapters=[];
                if(navigator.gpu){
                    const request=navigator.gpu.requestAdapter.bind(navigator.gpu);
                    navigator.gpu.requestAdapter=async(options={})=>{
                        const adapter=await request({...options,forceFallbackAdapter:true});
                        const info=adapter?.info;
                        window.audit_adapters.push(adapter?{
                            vendor:info?.vendor,architecture:info?.architecture,device:info?.device,
                            description:info?.description,isFallbackAdapter:adapter.isFallbackAdapter,
                            maxBufferSize:adapter.limits.maxBufferSize,
                            maxStorageBufferBindingSize:adapter.limits.maxStorageBufferBindingSize
                        }:null);
                        return adapter;
                    };
                }
            })();""")
            page=context.new_page()
            page.on('console',lambda message:result['console'].append(dict(type=message.type,text=message.text)))
            page.on('pageerror',lambda error:result['page_errors'].append(str(error)))
            print('Launching local official editor on isolated software browser',flush=True)
            page.goto(f'http://127.0.0.1:{port}/',wait_until='domcontentloaded',timeout=60000)
            page.wait_for_function("document.querySelector('#file-selector') || document.querySelector('#startup-error')",timeout=60000)
            result['adapters']=page.evaluate('window.audit_adapters')
            result['startup_text']=page.locator('body').inner_text()
            descriptions=' '.join(str(info) for info in result['adapters'])
            if not any(word in descriptions.lower() for word in ['swiftshader','software','llvmpipe']):
                raise ValueError('A CPU software WebGPU adapter was not verified; do not load model on another GPU')
            if page.locator('#startup-error').count():
                raise ValueError('Official editor failed to initialize: '+page.locator('#startup-error').inner_text())
            if args.ply:
                print('Importing actual local PLY, expected Gaussian count',result['expected_gaussians'],flush=True)
                page.locator('#file-selector').set_input_files(str(ply),timeout=60000)
                page.wait_for_function("expected => Number(document.querySelector('#status-bar .status-bar-stat-value')?.textContent.replace(/[^0-9]/g,''))===expected",
                    arg=result['expected_gaussians'],timeout=args.timeout_seconds*1000)
                result['displayed_gaussians']=int(''.join(c for c in page.locator('#status-bar .status-bar-stat-value').first.inner_text() if c.isdigit()))
                result['status']='actual_import_count_verified_pending_direct_visual_review'
                # Give the software compute sorter a chance to submit its
                # first scene frame; direct image inspection is still required.
                page.wait_for_timeout(10000)
                # Use the official editor's frame-selection shortcut. The
                # imported scene can otherwise sit partly outside the viewport.
                page.locator('body').click(position={'x': 600, 'y': 100})
                page.keyboard.press('f')
                page.wait_for_timeout(4000)
                result['camera_action']='Official camera.focus (F), selected imported model bounds; no model parameters edited'
            else:
                result['status']='official_editor_software_backend_ready'
            page.screenshot(path=str(output/'SUPERSPLAT_SCREENSHOT.png'))
            result['screenshot']=str(output/'SUPERSPLAT_SCREENSHOT.png')
            result['final_text']=page.locator('body').inner_text()
            result['canvas_count']=page.locator('canvas').count()
            result['browser_version']=browser.version
            browser.close()
    except Exception as error:
        result.update(status='browser_audit_failed',error=str(error))
    finally:
        server.shutdown();server.server_close()
        result['finished']=time.time();result['seconds']=result['finished']-result['started']
        write_json(output/'BROWSER_AUDIT.json',result)
    print(json.dumps({key:result.get(key) for key in ['status','displayed_gaussians','adapters','error','screenshot']},ensure_ascii=False),flush=True)
    if result['status']=='browser_audit_failed':
        raise SystemExit(1)


if __name__=='__main__':
    main()
