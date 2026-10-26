FROM python:3.12-slim
WORKDIR /work
ENV OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# Reproduces tests, experiments, analysis and the manuscript (.docx) inside the container.
CMD ["bash", "run_all.sh"]
