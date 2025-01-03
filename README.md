# AI Request Interface

This repository provides a Python-based library to manage various AI requests, including chat, audio, and image processing. It features a robust interface, error handling, and implementations leveraging GPT models for seamless AI-powered interactions.

---

## Features

- **Flexible Request Types**: Supports audio (transcriptions, speech), image (e.g., OCR), and chat requests.
- **Error Handling**: Includes custom exceptions for invalid or missing request types.
- **GPT Integration**: Implements GPT-based models for chat and generative tasks.
- **Extensible Design**: Built with modularity and scalability in mind, allowing easy integration of new request types.

---

## Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/sepehrtdarvish/AI-Request-Package.git
cd AI-Request-Package
pip install -r requirements.txt
```

---

## Usage

### Example: Creating a Chat Request

```python
from ai_request_interface import AIRequestInterface

# Initialize a chat request
data = {
    "messages": [{"role": "user", "content": "Hello, AI!"}],
    "temperature": 0.7
}
chat_request = AIRequestInterface(type="chat", **data)

# Send the request
response = chat_request.send()
print(response)
```

### Example: Handling Invalid Request Types

```python
from ai_request_interface import AIRequestInterface
from exceptions import RequestTypeNotValid

try:
    invalid_request = AIRequestInterface(type="invalid")
except RequestTypeNotValid as e:
    print(f"Error: {str(e)}")
```

---

## Project Structure

- **`ai_request_interface.py`**: Main interface for managing AI requests.
- **`exceptions.py`**: Custom exceptions for robust error handling.
- **`gpt_request.py`**: GPT-based request implementations and abstract base classes.

---

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests for bug fixes or feature enhancements.

---


## Contact

For questions or support, please contact [sepehrtdarvish@gmail.com].

