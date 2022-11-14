#Base Image to use
FROM python:3.10.8-slim

#Expose port 8080
EXPOSE 8080

#Optional - install git to fetch packages directly from github
RUN apt-get update && apt-get install -y git

#Copy all files in current directory into app directory
COPY . /app

#Change Working Directory to app directory
WORKDIR /app

#Run the application on port 8080
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]
