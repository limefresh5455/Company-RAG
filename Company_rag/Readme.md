# Chatbot Readme

This README file provides instructions for setting up and running the chatbot project.

## 1. Creating a New Python Environment

### For Linux/Mac:

```bash
# Create a new Python environment using virtualenv
pip install virtualenv
virtualenv env
source env/bin/activate
```

### For Windows:

```
# Create a new Python environment using virtualenv
pip install virtualenv
virtualenv env
.\env\Scripts\activate
```

## 2. Installing Requirements
```
pip install -r requirements.txt
```

## 3. Running database.py
this will crete a faiss vector database to test chatbot
```
python database.py
```

## 4. Running main.py
Finally, you can run the main script to start the chatbot server
```
python main.py
```