PY ?= python

.PHONY: data train test study validate report web all
data:      ; $(PY) -m spred build-data
train:     ; $(PY) -m spred train
test:      ; $(PY) -m pytest tests -q
study:     ; $(PY) scripts/methodology.py
validate:  ; $(PY) scripts/validation.py
report:    ; cd report && ./generate_report.sh substrate_predictor_report
web:       ; $(PY) web/make_fixture.py && node web/test_engine.js && $(PY) web/build_tool.py
all: data train test study validate
