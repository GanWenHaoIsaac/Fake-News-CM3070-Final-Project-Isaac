# Ensure this system is ran in accordance with the Conda Virtual Environments settings listed in the requirements.txt for both CPU and GPU devices
This code should be run on Python 3.10.13, with the installations from "requirementsCPU.txt" ot

## The 'MAIN' branch version of this assignment is incomplete, please use the one in the 'MASTER' branch instead. 

## Additionally, use "git lfs pull" then git clone this repository to ensure the BERT models load correctly. Download the ZIP will causes those models to not load. The report will link a Google Drive to this project if the methods above do not work.

Ensure you also use 'npm install' before doing any of the below tasks

How to run both backend and front end applications:
> Backend:  
- cd backend
- python app.py (for running with the frontend application OR "python flask_test_app" for backend only)
- open 'http://127.0.0.1:5000' on your browser

> Frontend: 
1. open a new terminal after running app.py
- npm run dev
- open 'http://localhost:5173/' on your browser


> How to run backend tests:
- cd backend
- pytest tests/test-app.py

> How to run frontend tests:
- npm test
