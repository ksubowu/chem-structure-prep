"""Continue dataset assembly and audit automatically after rendering completes."""
from runtime import ROOT
import time,subprocess,sys,json
while not (ROOT/'generation_manifest.json').exists():time.sleep(5)
for script,log in [('finalize.py','finalize.log'),('validate_data.py','validation.log')]:
 with (ROOT/log).open('w') as handle:
  result=subprocess.run([sys.executable,str(ROOT/script)],stdout=handle,stderr=subprocess.STDOUT)
 if result.returncode:
  (ROOT/'pipeline_status.json').write_text(json.dumps({'failed_stage':script,'returncode':result.returncode}));raise SystemExit(result.returncode)
(ROOT/'pipeline_status.json').write_text(json.dumps({'status':'assembled_and_validated'}))
