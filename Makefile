env := kaggle
kernel_name := kaggle
kernel_display_name := Python (kaggle)

.PHONY: setup update kernel lab clean

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

clean:
	find . -type d \( -name '.pytest_cache' -o -name '.kaggle_kernels' -o -name '.ipynb_checkpoints' -o -name '__pycache__' \) -prune -exec rm -rf {} +
	find . -type f \( -name '*.pyc' -o -name '.DS_Store' \) -delete
	@for dir in competitions/*/data/raw competitions/*/data/processed competitions/*/submissions; do \
		if [ -d "$$dir" ]; then \
			find "$$dir" -mindepth 1 ! -name '.gitkeep' -exec rm -rf {} +; \
		fi; \
	done
	@for dir in competitions/*/models; do \
		if [ -d "$$dir" ]; then \
			find "$$dir" -mindepth 1 -type f ! -name '*.py' ! -name '.gitkeep' -delete; \
			find "$$dir" -depth -mindepth 1 -type d -empty -delete; \
		fi; \
	done
