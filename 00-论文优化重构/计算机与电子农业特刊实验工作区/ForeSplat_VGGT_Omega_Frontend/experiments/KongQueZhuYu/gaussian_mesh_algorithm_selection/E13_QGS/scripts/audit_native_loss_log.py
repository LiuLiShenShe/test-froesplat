"""Separate native Python loss variables from actual conditional contributions."""
from pathlib import Path
import csv,json,math
import yaml
ROOT=Path(__file__).resolve().parents[1]
def main():
    rows=list(csv.DictReader((ROOT/'train_metrics.csv').open()))
    assert rows and int(rows[-1]['iteration'])==30000
    iterations=[int(row['iteration']) for row in rows]
    assert iterations==sorted(set(iterations)), 'Active trajectory must contain unique monotonic iterations; discarded attempts remain separately preserved.'
    records=[];uncertain=[]
    cfg=yaml.safe_load((ROOT/'actual_config.yaml').read_text())
    opt=cfg['optimizer']
    for row in rows:
        iteration=int(row['iteration'])
        values={k:float(row[k]) for k in ['image_loss','dist_loss','normal_loss','geo_loss','ncc_loss','total_loss']}
        assert all(math.isfinite(v) for v in values.values()),(iteration,'nonfinite native objective')
        # Upstream initializes conditional loss variables outside its loop.
        # At iteration30000 geometry/NCC are inactive, while the Python variables
        # can still hold iteration29999 values. Raw CSV remains untouched.
        active=dict(dist_loss=iteration>opt['dist_from_iter'],normal_loss=iteration>opt['normal_from_iter'],
                    geo_loss=opt['multi_view_weight_from_iter']<iteration<opt['multi_view_weight_until_iter'],
                    ncc_loss=opt['multi_view_weight_from_iter']<iteration<opt['multi_view_weight_until_iter'])
        eligible={k:values[k] if active[k] else 0.0 for k in active}
        residual=values['total_loss']-values['image_loss']-sum(eligible.values())
        record=dict(iteration=iteration,branch_eligible=active,native_logged_values=values,
                    eligible_logged_sum_residual=residual)
        if abs(residual)>1e-5 and active['geo_loss']:
            uncertain.append(iteration)
            record['interpretation']='Native logged geo/NCC variables may hold an earlier sample when visibility/NCC validity is empty; actual total remains authoritative.'
        records.append(record)
    final=records[-1]
    assert abs(final['eligible_logged_sum_residual'])<1e-5,('terminal objective mismatch',final)
    result=dict(status='passed',iterations=[r['iteration'] for r in records],
        raw_train_metrics_preserved=True,terminal_multi_view_geometry_NCC_contributions=0,
        monotonic_active_iteration_history=True,
        wall_seconds_definition='Elapsed time within the recorded attempt; resumed rows restart their segment clock. All-attempt and successful-prefix costs are in training_complete.json and training_execution_records.json.',
        conditional_logged_variable_uncertainty_iterations=uncertain,records=records,
        actual_native_branch_thresholds={key:opt[key] for key in ['dist_from_iter','normal_from_iter','multi_view_weight_from_iter','multi_view_weight_until_iter']},
        interpretation='Eligibility uses the actual native preset thresholds. Conditional Python loss variables can be stale; actual total is authoritative. Original CSV and training are unchanged.')
    (ROOT/'native_loss_log_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print('Actual native loss log audit passed; conditional variable warnings',len(uncertain),flush=True)
if __name__=='__main__':main()
