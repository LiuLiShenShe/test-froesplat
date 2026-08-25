import struct, os, json

base = r'D:\CAAS\04-COLMAP'
folders = ['BaiZhang','CaoMei1','CaoMei2','ChangShouHua1','ChangShouHua2','ChangShouHua3',
           'DouBanLv1','DouBanLv2','DouBanLv3','HongZhang','KongQueZhuYu',
           'WanNianQing1','WanNianQing2','WangWenCao1','WangWenCao2',
           'XianKeLai1','XianKeLai2','XianKeLai3','XiangPiShu1','XiangPiShu2']

results = {}
print(f"{'Folder':<18s} {'Input':>5s} {'Reg':>5s} {'Rate':>7s} {'3DPts':>8s} {'Status'}")
print("-" * 65)

for f in folders:
    sp = os.path.join(base, f, 'sparse', '0')
    inp = os.path.join(base, f, 'input')
    img = os.path.join(base, f, 'images')
    
    if not os.path.exists(sp):
        print(f"{f:<18s}   -- sparse/0 not found (still running?)")
        results[f] = {"status": "missing"}
        continue
    
    with open(os.path.join(sp, 'images.bin'), 'rb') as fh:
        ni = struct.unpack('<Q', fh.read(8))[0]
    with open(os.path.join(sp, 'points3D.bin'), 'rb') as fh:
        np3 = struct.unpack('<Q', fh.read(8))[0]
    
    ic = len([x for x in os.listdir(inp) if os.path.isfile(os.path.join(inp, x))])
    uc = len([x for x in os.listdir(img) if os.path.isfile(os.path.join(img, x))]) if os.path.exists(img) else 0
    rate = ni / ic * 100 if ic > 0 else 0
    
    if rate >= 70:
        st = "OK"
    elif rate >= 30:
        st = "WARN"
    else:
        st = "FAIL"
    
    print(f"{f:<18s} {ic:>5d} {ni:>5d} {rate:>6.1f}% {np3:>8d} {st}")
    results[f] = {"input": ic, "registered": ni, "rate": round(rate, 1), "points3d": np3, "status": st}

print("\n=== Summary ===")
ok = sum(1 for v in results.values() if v.get("status") == "OK")
warn = sum(1 for v in results.values() if v.get("status") == "WARN")
fail = sum(1 for v in results.values() if v.get("status") == "FAIL")
miss = sum(1 for v in results.values() if v.get("status") == "missing")
print(f"OK (>=70%): {ok}")
print(f"WARN (30-70%): {warn}")
print(f"FAIL (<30%): {fail}")
if miss > 0:
    print(f"Missing: {miss}")

print("\nFolders needing re-processing:")
for f, v in results.items():
    if v.get("status") in ("FAIL", "WARN"):
        print(f"  {f}: {v.get('rate', '?')}% ({v['status']})")
