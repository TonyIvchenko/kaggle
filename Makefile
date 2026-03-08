env := kaggle
kernel_name := kaggle
kernel_display_name := Python (kaggle)

.PHONY: setup update kernel lab

setup: environment.yml
	conda env create -f environment.yml
	$(MAKE) kernel

update: environment.yml
	conda env update -f environment.yml --prune
	$(MAKE) kernel

kernel:
	@if conda run -n $(env) jupyter kernelspec list --json | python -c 'import json,sys; sys.exit(0 if "$(kernel_name)" in json.load(sys.stdin).get("kernelspecs", {}) else 1)'; then \
		conda run -n $(env) jupyter kernelspec remove -f $(kernel_name); \
	else \
		echo "Kernel $(kernel_name) is not installed yet; skipping removal."; \
	fi
	conda run -n $(env) python -m ipykernel install --user --name $(kernel_name) --display-name "$(kernel_display_name)"

lab:
	conda run -n $(env) jupyter lab
