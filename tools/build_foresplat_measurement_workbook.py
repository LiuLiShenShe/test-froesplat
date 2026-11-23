#!/usr/bin/env python3
"""Build a blank-input measurement workbook from the completed ForeSplat runs."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter as col
from openpyxl.worksheet.datavalidation import DataValidation


ROOT = Path(__file__).resolve().parents[1]
POTS = ROOT / 'experiments/ForeSplat_MultiSpecies_Phenotyping'
TOMATO = ROOT / 'experiments/ForeSplat_TomatoPGT_Independent'
BENCHMARK = Path('/fj/end-paper/tomato_3d_benchmark')
HISTORY = ROOT / '植株数据.xlsx'
POT, TOM = '盆栽', '番茄'


def rows(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def tomato_reference(scene):
    return BENCHMARK / '03_standardized/TomatoPGT' / scene / 'annotations'


def plant_rows(potted, metric):
    for dataset, samples, id_key, group_key, run in (
        (POT, potted, 'plant_id', 'group', POTS),
        (TOM, metric, 'scene', 'cultivar', TOMATO),
    ):
        for row in samples:
            scene = row[id_key]
            model = run / scene
            ref_cloud = tomato_reference(scene) / 'reference_cloud/official_point_cloud.ply'
            source = model / 'calibration/scale_calibration.json'
            if dataset == TOM and (model / 'summary/official_reference.json').is_file():
                source = model / 'summary/official_reference.json'
            yield [dataset, scene, row[group_key]], [
                model / 'plant_only/plant_only_cm.ply',
                HISTORY if dataset == POT else ref_cloud,
                model / 'calibration/reference_frame.json',
                model / 'summary/rgb.png',
                model / 'phenotype/plant_measurements.csv',
                source,
            ]


def leaf_rows(matches, compounds, prefill):
    for row in matches:
        model = Path(row['model_local_path'])
        yield [POT, row['plant_id'], row['handwritten_number'], row['leaf_id'], '57 片人工确认', '', ''], [
            model, HISTORY, None, Path(row['original_rgb_local_path']), Path(row['leaf_reference_local_path']),
            None, None, None, None, None,
        ]
    for row in compounds:
        scene, label = row['scene'], row['official_point_instance']
        candidate = row['candidate_id']
        model = Path(row['model_local_path']) if row['model_local_path'] else None
        suggestion = prefill.get((scene, label), {})
        figure = TOMATO / 'summary/compound_auto_review' / scene / row['overlay'] if row['overlay'] else None
        yield [TOM, scene, f'点标签 {label}', candidate, row['status'], row['official_id'], row['metric_pair_qc']], [
            model, tomato_reference(scene) / 'reference_cloud/official_point_cloud.ply',
            BENCHMARK / '04_annotations/instances/TomatoPGT' / f'{scene}_labels.csv',
            model.with_name('reference_rgb.png') if model else None, figure,
            *(float(suggestion[k]) if suggestion.get(k) else None for k in (
                'model_chord_cm', 'official_chord_cm', 'model_transverse_width_cm', 'official_transverse_width_cm',
            )),
            suggestion.get('endpoints_json'),
        ]


def format_sheet(sheet):
    sheet.freeze_panes = 'C2'
    sheet.auto_filter.ref = sheet.dimensions
    sheet.row_dimensions[1].height = 32
    for cell in sheet[1]:
        cell.fill = PatternFill('solid', fgColor='155545')
        cell.font = Font(color='FFFFFF', bold=True)
        cell.alignment = Alignment(wrap_text=True, vertical='center')
    for cells in sheet.columns:
        width = max(len(str(c.value or '')) for c in cells if c.data_type != 'f')
        sheet.column_dimensions[cells[0].column_letter].width = max(13, min(42, width + 3))


def measurement_sheet(book, name, identities, traits, extra, records):
    sheet = book.create_sheet(name)
    sheet.append(identities + [f'{trait} {key}' for trait in traits for key in ('模型值', '参考值', '质控', '差值')]
                 + ['参考来源 / Excel行 / 端点证据', '质控备注'] + extra)
    groups = defaultdict(list)
    first = len(identities) + 1
    model_path_col = col(first + 4 * len(traits) + 2)
    for row_num, (identity, details) in enumerate(records, 2):
        groups[identity[0]].append(row_num)
        sheet.append(identity + [None] * (4 * len(traits) + 2) + [str(v) if isinstance(v, Path) else v for v in details])
        for cell, value in zip(sheet[row_num][-len(details):], details):
            if isinstance(value, Path) and value.is_file():
                cell.hyperlink = value.as_uri()
                cell.style = 'Hyperlink'
        for start in range(first, first + 4 * len(traits), 4):
            pred, ref, qc = (f'{col(j)}{row_num}' for j in range(start, start + 3))
            sheet.cell(row_num, start + 3, (
                f'=IF(AND({qc}="通过",{model_path_col}{row_num}<>"",COUNT({pred}:{ref})=2,'
                f'{pred}>0,{ref}>0),{pred}-{ref},"")'
            )).number_format = '0.000'
        for j in range(first, first + 4 * len(traits) + 2):
            cell = sheet.cell(row_num, j)
            if cell.data_type != 'f':
                cell.fill = PatternFill('solid', fgColor='E7F3EB')
    format_sheet(sheet)
    validation = DataValidation(type='list', formula1='"通过,待核,不纳入"', allow_blank=True, showErrorMessage=True)
    validation.error = '请选择 通过、待核 或 不纳入。'
    sheet.add_data_validation(validation)
    for j in range(first + 2, first + 4 * len(traits), 4):
        validation.add(f'{col(j)}2:{col(j)}{sheet.max_row}')
    for dataset, rr in groups.items():
        for i, trait in enumerate(traits):
            letter = col(first + 4 * i + 3)
            yield dataset, trait, len(rr), f"'{name}'!{letter}{rr[0]}:{letter}{rr[-1]}"


def build(output):
    potted = rows(POTS / 'summary/all19_results.csv')
    metric = [r for r in rows(TOMATO / 'summary/batch_results.csv') if r['metric_mesh'] == 'True']
    matches = [r for r in rows(POTS / 'summary/three_leaf_model_locations.csv') if r['human_confirmed'] == 'True']
    review = TOMATO / 'summary/compound_auto_review'
    compounds = [r for r in rows(review / 'compound_match_suggestions.csv')
                 if r['metric_scan'] == 'True' and r['reference_scope'] == 'compound_leaf']
    prefill = {(r['scene'], r['official_point_instance']): r for r in rows(review / 'provisional_leaf_measurements.csv')}

    book = Workbook()
    info = book.active
    info.title = '先读这里'
    for text in (
        '填写顺序：先从对应行打开模型和独立参考，再填写两侧的值；质控选“通过”后才计入汇总。',
        '整株：19 株盆栽、19 个番茄扫描可供公制测量。株高用可见竖直跨度；冠幅用最大水平跨度。',
        '叶片：盆栽 57 行是已确认身份；番茄保留 70 条官方复叶，其中 67 条优先，另有 2 条歧义、1 条无匹配。',
        '端点：同一叶基到同一叶尖的三维直线为叶长；同一截面左右叶缘的三维直线为叶宽。',
        '盆栽参考为已核对同株同叶的历史实测或可验证的同批实测。番茄参考为同一扫描的独立官方点云。',
        '官方原始点云请先按 CloudGraph 元数据确认坐标、尺度和竖直方向。ForeSplat 厘米 PLY 不乘任何比例。',
        'CH_07112025 有官方点云和标签，但无对应 CloudGraph 文件；先独立确认参考点云的单位和竖直再纳入误差。',
        '预填值仅供端点核验，缺项、歧义和 scale_review 先留空。单侧观测面积没有参考值时不计算误差。',
        '保留参考来源、Excel 行或官方标签、测量端点截图和操作者。界面中的图像复核与逐性状质控分开。',
        '误差定义为模型值减去参考值；汇总按数据集和性状分别算 n、MAE、RMSE 和平均偏差。',
        '路径列可点击服务器本机文件。外部机器用完整路径定位对应文件或保留目录结构下载。',
    ):
        info.append([text])
    info.column_dimensions['A'].width = 116
    for cells in info.rows:
        cells[0].alignment = Alignment(wrap_text=True, vertical='center')
        info.row_dimensions[cells[0].row].height = 28

    summary = book.create_sheet('误差汇总')
    summary.append(['数据集', '性状', '表中行数', '有效配对 n', 'MAE', 'RMSE', '平均偏差'])
    plant = measurement_sheet(book, '整株', ['数据集', '样本ID', '批次或品种'], ['株高 cm', '冠幅 cm'],
                              ['公制植株 PLY', '独立参考', '估计竖直方向', 'RGB', '初步模型测量', '尺度或官方来源'],
                              plant_rows(potted, metric))
    leaf = measurement_sheet(book, '叶片', ['数据集', '样本ID', '手写编号或点标签', '模型ID', '已有身份记录', '官方骨架边ID', '已有尺度检查'],
                             ['直线叶长 cm', '直线叶宽 cm', '单侧叶面积 cm²'],
                             ['叶片 PLY', '独立参考', '官方点标签文件', '原始 RGB', '掩膜或对照图', '初步模型叶长 cm', '初步参考叶长 cm',
                              '初步模型叶宽 cm', '初步参考叶宽 cm', '预填端点坐标'], leaf_rows(matches, compounds, prefill))
    for i, (dataset, trait, count, diff) in enumerate([*plant, *leaf], 2):
        summary.append([dataset, trait, count, f'=COUNT({diff})',
                        f'=IF(D{i}=0,"",(SUMIF({diff},">0")-SUMIF({diff},"<0"))/D{i})',
                        f'=IF(D{i}=0,"",SQRT(SUMSQ({diff})/D{i}))', f'=IF(D{i}=0,"",AVERAGE({diff}))'])
        for j in (5, 6, 7):
            summary.cell(i, j).number_format = '0.000'
    format_sheet(summary)
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite measurement inputs: {output}')
    book.save(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    build(parser.parse_args().output)
