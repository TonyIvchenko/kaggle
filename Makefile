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
	-conda run -n $(env) jupyter kernelspec remove -f $(kernel_name)
	conda run -n $(env) python -m ipykernel install --user --name $(kernel_name) --display-name "$(kernel_display_name)"

lab:
	conda run -n $(env) jupyter lab
