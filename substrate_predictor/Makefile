PY ?= python

.PHONY: data train test study validate report all
data:      ; $(PY) -m spred build-data
train:     ; $(PY) -m spred train
test:      ; $(PY) -m pytest tests -q
study:     ; $(PY) scripts/methodology.py
validate:  ; $(PY) scripts/validation.py
all: data train test study validate
