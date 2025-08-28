install:
	python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt

train:
	source venv/bin/activate && python3 src/train.py --model rf --n_estimators 200 --max_depth 5

ui:
	source venv/bin/activate && mlflow ui --backend-store-uri ./mlruns --port 5000

test:
	pytest -v

clean:
	rm -rf mlruns __pycache__ .pytest_cache
