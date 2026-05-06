Tristan Rhyl C. Penaso

Airah Nichole Montillano

Kylle Mae Mercado

Download the IS108-FINALPROJECT Folder then:

Open the CMD within the IS108-FINALPROJECT and create a virtual environment
A virtual environment keeps your project's libraries separate from other Python projects on your laptop.
To create a virtual environment, type and run this:


python -m venv venv


This creates a venv folder inside IS108_Project. You only do this once.

Activate the virtual environment
You must activate it every time you open a new Command Prompt window, run:


venv\Scripts\activate


When it's active, your prompt will show (venv) at the start of the line. That means you're good.
If you get a "cannot be loaded" error, first, run:


Set-ExecutionPolicy -Scope CurrentUser RemoteSigned 


in PowerShell, then try again in cmd.



Phase C — Install Libraries
Upgrade pip first
Always upgrade pip before installing packages to avoid dependency errors, run:


python -m pip install --upgrade pip


Installing all required libraries takes 2–5 min, please be patient
Paste this entire command, it installs everything the app needs in one shot, run:


pip install streamlit pandas numpy scikit-learn matplotlib seaborn openpyxl


You'll see a lot of text scrolling, that's normal. Wait until you see "Successfully installed..." at the end.
Verify all packages installed correctly. By running:


pip list


Scroll through the list and confirm you see: streamlit, pandas, numpy, scikit-learn, matplotlib, seaborn, openpyxl.
To confirm app.py is there, type: 


dir 


in your cmd window. You should see app.py listed.


Run the application
Make sure (venv) is still showing in your prompt, then run:


streamlit run app.py


Your browser will open automatically at http://localhost:8501  that's the app. Leave the cmd window open while using it.


Every time you come back to it
You don't need to reinstall anything. Open the cmd of the folder and just do these two steps each session:


venv\Scripts\activate

streamlit run app.py
