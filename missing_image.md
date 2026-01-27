 - Using cached token for subAccount 'SFMobileSH'.
2025-12-30 12:36:29,377 - INFO - Forwarding request to https://api.ai.intprod-eu12.eu-central-1.aws.ml.hana.ondemand.com/v2/inference/deployments/df4d05d1a5806334/converse-stream with subAccount 'SFMobileSH'
2025-12-30 12:36:29,378 - INFO - Raw request received (streaming): {
  "model": "claude-4.5-sonnet",
  "temperature": 0.1,
  "reasoning_effort": "medium",
  "messages": [
    {
      "role": "user",
      "content": "Hi"
    },
    {
      "role": "assistant",
      "content": "Hi! How can I help you today? If you share a bit about what you need\u2014a question, a task, or an idea\u2014I\u2019ll jump right in."
    },
    {
      "role": "user",
      "content": [
        {
          "type": "image_url",
          "image_url": {
            "url": "data:image/png;base64,dsr23114214234"
        }
      ]
    }
  ],
  "stream": true,
  "stream_options": {
    "include_usage": true
  }
}
2025-12-30 12:36:29,378 - INFO - Forwarding payload to API (streaming): {
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "text": "Hi"
        }
      ]
    },
    {
      "role": "assistant",
      "content": [
        {
          "text": "Hi! How can I help you today? If you share a bit about what you need\u2014a question, a task, or an idea\u2014I\u2019ll jump right in."
        }
      ]
    }
  ],
  "inferenceConfig": {
    "temperature": 0.1
  }
}
2025-12-30 12:36:33,578 - INFO - Using Claude 3.7/4 streaming for subAccount 'SFMobileSH'
