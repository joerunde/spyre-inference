# Under COVERAGE=1 the Makefile puts this dir on PYTHONPATH so every interpreter
# (pytest + spawned vLLM workers) starts coverage. process_startup() is a no-op
# unless COVERAGE_PROCESS_START is set, so this is inert otherwise. The bare
# except is load-bearing: a raise here would break every interpreter, not just
# coverage.
try:
    import coverage

    coverage.process_startup()
except Exception:
    pass
