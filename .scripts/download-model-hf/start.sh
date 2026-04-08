export AWS_ACCESS_KEY_ID=123
export AWS_SECRET_ACCESS_KEY=1223
export HF_TOKEN=hf_1212

python3 hf_to_s3.py \
  --model-id speakleash/Bielik-11B-v3-Base-20250730 \
  --bucket models-062f015a-ce33-4f0b-b33c-af2505cfd4da \
  --endpoint-url "https://s3.openshift-storage.svc:443" \
  --cache-dir bielik_cache