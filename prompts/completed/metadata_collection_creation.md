# Collections endpoint needs to have metadata automatically sent and the name of the avatar needs to be automatically preprocessed so that the name works every time.
/collections

{
  "name": "julia_ann",
  "metadata": {"topic": "Avatar", "version": "1.0"}
}

curl -X 'POST' \
  'http://localhost:8088/collections/' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "name": "julia_ann",
  "metadata": {"topic": "AI", "version": "1.0"}
}'



{
  "detail": "Failed to create collection: 1 validation error for CreateCollectionResponse\nid\n  Input should be a valid string [type=string_type, input_value=UUID('3db13093-4693-4bf6-a58a-08183cef7862'), input_type=UUID]\n    For further information visit https://errors.pydantic.dev/2.5/v/string_type"
}
