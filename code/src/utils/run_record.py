"""Failure-aware single-run records. Does not suppress or relabel software errors."""
from datetime import datetime, timezone


def execute(config, runner):
    context, provenance, completed = {}, {}, []
    current_round = None

    def observe(event, payload):
        nonlocal current_round
        if event == 'provenance':
            provenance.update(payload)
            return
        if 'round' in payload:
            current_round = payload['round']
        context.clear()
        context.update(event=event,round=current_round,**{k:v for k,v in payload.items() if k!='round'})
        if event == 'round_end':
            completed.append(dict(payload))

    record={'config':config.to_dict(), 'started_utc':datetime.now(timezone.utc).isoformat()}
    try:
        result=runner(config,verbose=False,on_event=observe)
        record.update(status='completed',result=result)
    except FloatingPointError as exc:
        record.update(status='numerical_failure',error=str(exc),final=None)
    except KeyboardInterrupt:
        record.update(status='interrupted',error='User interruption',final=None)
    except Exception as exc:
        record.update(status='execution_error',error=f'{type(exc).__name__}: {exc}',final=None)
    record.update(provenance=provenance,completed_rounds=completed,
                  completed_round_count=len(completed),last_context=dict(context),
                  finished_utc=datetime.now(timezone.utc).isoformat())
    return record
