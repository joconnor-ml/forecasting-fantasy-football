#Base Image to use
FROM python:3.10.8-slim

#Expose port 8080
EXPOSE 8080

# Upgrade pip and install requirements
COPY requirements.txt requirements.txt
RUN pip install -U pip
RUN pip install -r requirements.txt

#Copy all files in current directory into app directory
COPY . /app

#Change Working Directory to app directory
WORKDIR /app

#Run the application on port 8080
ENTRYPOINT ["find", "/usr/local/lib/python3.9/site-packages/streamlit", "-type", "f", "\( -iname \*.py -o -iname \*.js \)", "-print0", "|", "xargs", "-0", "sed", "-i", "'s/healthz/health-check/g'", "&&", "streamlit", "run", "Forecasting_Fantasy_Football.py", "--server.enableCORS", "false", "--server.enableXsrfProtection", "false", "--server.port=8080", "--server.address=0.0.0.0"]
