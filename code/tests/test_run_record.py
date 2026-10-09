import json
import unittest
from src.experiment_config import RunConfig
from src.utils.run_record import execute


class RunRecord(unittest.TestCase):
    def test_failed_outcomes_keep_provenance_and_never_fake_metrics(self):
        for error,status in [(FloatingPointError('bad loss'),'numerical_failure'),
                             (ValueError('invalid privacy history'),'execution_error'),
                             (KeyboardInterrupt(),'interrupted')]:
            def runner(cfg,verbose,on_event):
                on_event('provenance',{'source_fingerprint':'test-code'})
                on_event('round_end',{'round':1,'accuracy':.5,'loss':1.})
                on_event('round_start',{'round':2})
                on_event('client_start',{'client':3})
                raise error
            record=execute(RunConfig(),runner)
            self.assertEqual(record['status'],status)
            self.assertEqual(record['provenance']['source_fingerprint'],'test-code')
            self.assertEqual(record['last_context']['round'],2)
            self.assertEqual(record['last_context']['client'],3)
            self.assertEqual(record['completed_round_count'],1)
            self.assertIsNone(record['final'])
            self.assertNotIn('result',record)
            json.dumps(record,allow_nan=False)

    def test_completed_record_preserves_result(self):
        record=execute(RunConfig(),lambda cfg,**kw: {'final':{'accuracy':.75}})
        self.assertEqual(record['status'],'completed')
        self.assertEqual(record['result']['final']['accuracy'],.75)


if __name__=='__main__': unittest.main()
