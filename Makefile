.PHONY: setup data train mlflow api test docker-build docker-run docker-stop

setup:
	pip install -r requirements.txt

data:
	python data/prepare_data.py

train:
	python models/train.py

mlflow:
	python mlflow_server/run_server.py

api:
	python app/api.py

test:
	python -m pytest tests/

docker-build:
	docker build -t imdb-sentiment-classifier .

docker-run:
	docker run -p 5000:5000 -p 5001:5001 -v $(PWD)/data:/app/data -v $(PWD)/mlruns:/app/mlruns -v $(PWD)/models/artifacts:/app/models/artifacts imdb-sentiment-classifier

docker-stop:
	docker stop $$(docker ps -q --filter ancestor=imdb-sentiment-classifier)

all: setup data train api 