"""Refresh aggregate statistics from persisted per-pair measurements, no data reread."""
from geometry_common import *
from geometry_multiview_audit import summarize

def main():
    rows=[]
    for path in sorted((AUDIT/'geometry_comparison').glob('multiview_*_pairs.csv')):
        with path.open() as f:
            for record in csv.DictReader(f):
                for key,value in list(record.items()):
                    if key in ['method','source','target','region','control']:continue
                    if value=='':record[key]=None
                    elif key in ['source_offset','queries','source_valid','projected_inside','target_overlap','source_zbuffer_occluded','target_predicted_occluded','supported_2pct','supported_5pct','supported_10pct','visible_candidates']:record[key]=int(value)
                    else:record[key]=float(value)
                rows.append(record)
    summary=summarize(rows)
    write_csv(AUDIT/'geometry_comparison/multiview_summary.csv',summary);write_json(AUDIT/'geometry_comparison/multiview_summary.json',summary)
    print('Refreshed saved pair summaries',len(rows),len(summary),'zero-query ROI pairs explicitly counted, excluded from per-pair quantiles')

if __name__=='__main__':main()
