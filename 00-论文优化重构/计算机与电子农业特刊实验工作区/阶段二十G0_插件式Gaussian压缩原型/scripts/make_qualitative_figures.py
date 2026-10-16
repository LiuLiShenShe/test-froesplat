"""Create small qualitative sheets from the frozen held-out render cache."""
from common import *
import csv


def read_rows(path):
    with Path(path).open(newline='', encoding='utf-8-sig') as handle:
        return list(csv.DictReader(handle))


def main():
    import numpy as np
    import torch
    from PIL import Image
    import matplotlib.pyplot as plt

    p = plan()
    reference = {r['camera']: r for r in read_rows(STAGE / 'configs/HELDOUT_CAMERA_REFERENCE_INDEX.csv')}
    render_index = {(r['run_id'], r['camera']): r for r in read_rows(STAGE / 'configs/FLOAT_RENDER_INDEX.csv')}
    metric_rows = {(r['run_id'], r['camera']): r for r in read_rows(STAGE / 'reports/HELDOUT_PER_VIEW_METRICS.csv')}
    runs = ['random_p00', 'random_p40', 'opacity_p40', 'random_p60', 'opacity_p60']
    output = STAGE / 'figures/qualitative'
    output.mkdir(parents=True, exist_ok=True)
    observations = []

    def load_rgb(path):
        image = Image.open(path).convert('RGB')
        return np.asarray(image, dtype=np.float32) / 255.0

    def load_render(run_id, camera):
        tensor = torch.load(render_index[(run_id, camera)]['float_render_path'],
                            map_location='cpu', weights_only=True)
        return tensor['render'].permute(1, 2, 0).numpy().clip(0.0, 1.0), tensor['rend_alpha'][0].numpy()

    for camera in p['qualitative_camera_names']:
        ref_row = reference[camera]
        ref = load_rgb(ref_row['reference_rgb_path'])
        roi = np.asarray(Image.open(ref_row['ROI_path']).convert('L')) > 127
        probe, _ = load_render(runs[1], camera)
        if ref.shape[:2] != probe.shape[:2]:
            target_size = (probe.shape[1], probe.shape[0])
            ref = np.asarray(Image.fromarray((ref * 255.0).astype(np.uint8)).resize(
                target_size, Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
            roi = np.asarray(Image.fromarray((roi.astype(np.uint8) * 255)).resize(
                target_size, Image.Resampling.NEAREST)) > 127
        ys, xs = np.where(roi)
        if len(xs) == 0:
            raise RuntimeError(f'empty qualitative ROI: {camera}')
        margin_x = max(12, int((xs.max() - xs.min() + 1) * 0.08))
        margin_y = max(12, int((ys.max() - ys.min() + 1) * 0.08))
        x0, x1 = max(0, xs.min() - margin_x), min(ref.shape[1], xs.max() + margin_x + 1)
        y0, y1 = max(0, ys.min() - margin_y), min(ref.shape[0], ys.max() + margin_y + 1)
        images = [ref]
        labels = ['reference']
        metrics = []
        for run_id in runs[1:]:
            image, _ = load_render(run_id, camera)
            images.append(image)
            labels.append(run_id)
            row = metric_rows[(run_id, camera)]
            metrics.append({'run_id': run_id,
                            'foreground_PSNR': float(row['foreground_PSNR']),
                            'proxy_inner_edge_PSNR': float(row['proxy_inner_edge_PSNR']),
                            'alpha_coverage': float(row['alpha_coverage'])})

        figure, axes = plt.subplots(2, len(images), figsize=(14, 8), constrained_layout=True)
        for col, (label, image) in enumerate(zip(labels, images)):
            axes[0, col].imshow(image)
            axes[0, col].set_title(label)
            axes[0, col].axis('off')
            crop = image[y0:y1, x0:x1]
            axes[1, col].imshow(crop)
            axes[1, col].axis('off')
            if col == 0:
                axes[1, col].set_ylabel('ROI crop')
        figure.suptitle(f'{camera}: reference and fixed-capacity RGB views')
        figure.savefig(output / f'{camera}_rgb_roi.png', dpi=130)
        plt.close(figure)

        error_figure, error_axes = plt.subplots(1, len(images) - 1, figsize=(14, 3.5), constrained_layout=True)
        for axis, run_id, image in zip(error_axes, runs[1:], images[1:]):
            error = np.abs(image - ref).mean(axis=2)
            axis.imshow(error, cmap='magma', vmin=0.0, vmax=0.35)
            axis.set_title(run_id)
            axis.axis('off')
        error_figure.suptitle(f'{camera}: absolute RGB error against frozen reference')
        error_figure.savefig(output / f'{camera}_error.png', dpi=130)
        plt.close(error_figure)

        observations.append({'camera': camera, 'roi_bbox_xyxy': [int(x0), int(y0), int(x1), int(y1)],
                             'reference_original_shape_hw': [int(Image.open(ref_row['reference_rgb_path']).height),
                                                             int(Image.open(ref_row['reference_rgb_path']).width)],
                             'evaluation_shape_hw': [int(ref.shape[0]), int(ref.shape[1])],
                             'reference_sha256': ref_row['reference_file_sha256'], 'metrics': metrics,
                             'rgb_figure': str(output / f'{camera}_rgb_roi.png'),
                             'error_figure': str(output / f'{camera}_error.png')})

    write_json(STAGE / 'reports/QUALITATIVE_OBSERVATIONS.json', {
        'status': 'PASS',
        'cameras': p['qualitative_camera_names'],
        'runs': runs,
        'roi_is_independent_GT': False,
        'interpretation': 'visual and fixed-ROI diagnostic only; no independent geometric ground truth',
        'observations': observations,
    })
    print('QUALITATIVE_FIGURES_PASS', len(observations), flush=True)


if __name__ == '__main__':
    main()
