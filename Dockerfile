#Sets the language as python 3.11.
#Slim means it gets rid of unecessary tools or libraries to keep the image small and efficient
FROM python:3.11-slim
#Sets the working directory inside the container to app
WORKDIR /app
#Copies everything from the current project directory to the /app directory inside the container
COPY . /app

COPY requirements.txt .
#Executes commands to install dependencies inside the container
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
#Default flask port
EXPOSE 5000
#Tells flask the entry point of the app
ENV FLASK_APP=app.py
#Tells Flask to run on all available network interfaces inside the container
#making it accessible to your local machine via the mapped port.
ENV FLASK_RUN_HOST=0.0.0.0

#Specifies command to run when the application starts
CMD ["flask", "run", "--host=0.0.0.0", "--port=5000"]
