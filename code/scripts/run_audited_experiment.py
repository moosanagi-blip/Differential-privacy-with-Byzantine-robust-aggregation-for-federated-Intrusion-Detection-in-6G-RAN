"""Validate an explicit config, or execute ONE run into a new output directory.

No training without --execute. This is not approval to launch a campaign.
Hard process termination can leave RUNNING.json without a final record; never
interpret that marker or a directory's existence as successful completion.
"""
import argparse
import json
import sys
from pathlib import Path

CODE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(CODE))
from src.experiment_config import RunConfig
from src.fl.simulation import run_federated
from src.utils.run_record import execute


def read_config(path):
    raw=json.loads(Path(path).read_text(encoding='utf-8-sig'))
    if not isinstance(raw,dict):
        raise ValueError('Config must be a JSON object')
    unknown=set(raw)-set(RunConfig.__dataclass_fields__)
    if unknown: raise ValueError(f'Unknown config keys: {sorted(unknown)}')
    cfg=RunConfig.from_dict(raw)
    if cfg.engine!='sequential': raise ValueError('Audited runner requires sequential engine')
    if not cfg.data_root: raise ValueError('An explicit dataset root is required')
    if cfg.dp and cfg.dp_calibrate!='worst': raise ValueError('DP requires worst-client calibration')
    return cfg


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--out',help='New run directory; must not already exist')
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    cfg=read_config(args.config)
    if not args.execute:
        print('Config parsed; training NOT started. Dataset and scientific protocol are not validated by this check.')
        print(json.dumps(cfg.to_dict(),indent=2))
        return
    if not args.out: parser.error('--out is required with --execute')
    out=Path(args.out)
    out.mkdir(parents=True,exist_ok=False)
    with (out/'RUNNING.json').open('x',encoding='utf-8') as stream:
        json.dump({'status':'started_not_completed','config':cfg.to_dict()},stream,indent=2)
    record=execute(cfg,run_federated)
    with (out/'record.json').open('x',encoding='utf-8') as stream:
        json.dump(record,stream,indent=2,allow_nan=False)
    print(f"{record['status']}: {out/'record.json'}")
    if record['status']!='completed': raise SystemExit(1)


if __name__=='__main__': main()
