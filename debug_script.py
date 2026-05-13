import sys
import trace
tracer = trace.Trace(count=False, trace=True, ignoredirs=[sys.prefix, sys.exec_prefix])
# tracer.run('import holosoma.holosoma.train_cvae')
