#!/usr/bin/env bash
# One-command local deployment: minikube cluster + API + monitoring.
# Usage (from repo root): bash scripts/deploy_local.sh
set -euo pipefail

echo "==> 1/6 Starting minikube"
minikube status >/dev/null 2>&1 || minikube start --driver=docker --cpus=3 --memory=6144
minikube addons enable ingress
minikube addons enable metrics-server

echo "==> 2/6 Building Docker image"
docker build -t heart-disease-api:1.0 .

echo "==> 3/6 Loading image into minikube"
minikube image load heart-disease-api:1.0

echo "==> 4/6 Deploying API (namespace, deployment, service, ingress, HPA)"
kubectl apply -f k8s/
kubectl rollout status deployment/heart-api -n heart-disease --timeout=180s

echo "==> 5/6 Installing Prometheus + Grafana"
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts >/dev/null 2>&1 || true
helm repo update >/dev/null
helm upgrade --install monitoring prometheus-community/kube-prometheus-stack \
  --namespace monitoring --create-namespace -f monitoring/prometheus-values.yaml --wait --timeout 10m
kubectl apply -f monitoring/servicemonitor.yaml
kubectl apply -f monitoring/grafana-dashboard.yaml

echo "==> 6/6 Smoke test"
curl -s --resolve "heart-api.local:80:$(minikube ip)" http://heart-api.local/health; echo
curl -s --resolve "heart-api.local:80:$(minikube ip)" -X POST http://heart-api.local/predict \
  -H "Content-Type: application/json" -d @sample_request.json; echo

cat <<EOF

Deployment complete.
  API (Ingress):  curl --resolve "heart-api.local:80:$(minikube ip)" http://heart-api.local/docs
  LoadBalancer:   run 'minikube tunnel' in another terminal, then: kubectl get svc -n heart-disease
  Prometheus:     kubectl port-forward -n monitoring svc/monitoring-kube-prometheus-prometheus 9090:9090
  Grafana:        kubectl port-forward -n monitoring svc/monitoring-grafana 3000:80   (admin / admin123)
  Traffic:        bash monitoring/generate_traffic.sh 300
EOF