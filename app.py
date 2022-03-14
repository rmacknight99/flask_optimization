import os
from flask import Flask, flash, request, redirect, url_for, send_from_directory, render_template
from werkzeug.utils import secure_filename
from flask_mail import Mail, Message
from scripts.make_simulator import make_dataset
from scripts.make_simulator import train_emulator
from scripts.make_campaign import optimize as opt
import shutil

UPLOAD_FOLDER = '' # folder to upload files into , default is in working directory
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'csv'} # allowed file extensions

app = Flask(__name__) # initialize flask app

#### app configuration for status of app and use of mail ####

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DEBUG'] = False
app.config['TESTING'] = False
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 465
app.config['MAIL_USE_TLS'] = False
app.config['MAIL_USE_SSL'] = True
app.config['MAIL_DEBUG'] = False
app.config['MAIL_USERNAME'] = 'rmacknight18@gmail.com'
app.config['MAIL_PASSWORD'] = 'weezer17'
app.config['MAIL_DEFAULT_SENDER'] = 'gomesgroup@datashare.com'
app.config['MAIL_MAX_EMAILS'] = 2
app.config['MAIL_SUPPRESS_SEND'] = False
app.config['MAIL_ASCII_ATTACHMENTS'] = True

mail = Mail(app) # initialize mail aspect of flask

def allowed_file(filename): # function for determining if the extension is allowed
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS # returns True or False if the extension is in the set of allowed extensions

@app.route('/')
def homepage():
        return render_template('homepage.html')

@app.route('/help')
def help_page():
        return render_template('help.html')

@app.route('/upload', methods=['GET', 'POST']) # define the route of the web app along with the methods associated with that path
def upload_file(): # define function to upload a file and email a file

        if request.method == 'POST': # if the method is POST meaning the user has acted on the route (in this case by uploading a file)

                if 'file' not in request.files: # check for file not being present
                        flash('No file part')
                        return redirect(request.url) # return to route

                file = request.files['file'] # get file if it is present
                if file.filename == '': # more checks
                        flash('No selected file')
                        return redirect(request.url)

                if file and allowed_file(file.filename): # if the file is present and the extension is allowed
                        filename = "data.csv" # change filenmame to data.csv
                        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename)) # save the file to the upload folder (current working directory)
                        send_from_directory(app.config["UPLOAD_FOLDER"], filename) # send the file from the directory to the machine the app is running from

                        if os.path.isdir("datasets/"):
                                pass
                        else:
                                os.mkdir("datasets/")

                        if os.path.isdir("datasets/dataset_custom"):
                                try:
                                        os.remove("datasets/dataset_custom/config.json")
                                        os.remove("datasets/dataset_custom/description.txt")
                                except:
                                        pass
                        else:
                                os.mkdir("datasets/dataset_custom")

                        if os.path.isdir("bank/"):
                                pass
                        else:
                                os.mkdir("bank/")

                        if os.path.isdir("campaigns/"):
                                pass
                        else:
                                os.mkdir("campaigns/")

                        shutil.move("{}".format(filename), "datasets/dataset_custom/data.csv") # move data into dataset directory

                        #### get variables from the user form (request.form.get()) ####

                                # THESE ARE THE EMULATOR VARIABLES #
                                
                        batch_size = request.form.get("batch_size")
                        reg = 0.001
                        hidden_act = request.form.get("h_act")
                        out_act = request.form.get("o_act")
                        max_epochs = request.form.get("epochs")
                        feature_transform = request.form.get("ft")
                        target_transform = request.form.get("tt")
                        save = request.form.get("save")

                        if save != '':
                                save = True
                        else:
                                save = False
                        
                                # THESE ARE THE OPTIMIZATION VARIABLES #

                        emulate = request.form.get("emulate")
                        optimize = request.form.get("optimize")
                        algorithm = request.form.get("algorithm")
                        goal = request.form.get("goal")
                        max_iter = request.form.get("max_iter")
                        campaign_number = request.form.get("campaign_number")

                        ###############################################################
                        
                        if emulate == optimize:
                                
                                if emulate == 'True': # if the user decides to train a model and suggest new (labeled) observations 
                                        name = 'custom'
                                        bnn,emulator,dataset,scores = train_emulator(name,int(batch_size),reg,str(hidden_act),str(out_act),int(max_epochs),str(feature_transform),str(target_transform),save=save)
                                        # make dataset and train emulator
                                        opt(dataset,name,str(algorithm),str(goal),int(max_iter),int(campaign_number),emulator=True)
                                        # run optimization campaign on dataset

                                else: # no option was selected

                                        print('no option selected')

                        elif emulate == 'True': # what to do if the user would like to build a model

                                name = 'custom'
                                bnn,emulator,dataset,scores = train_emulator(name,int(batch_size),reg,str(hidden_act),str(out_act),int(max_epochs),str(feature_transform),str(target_transform),save=save)
                                # make dataset and train emulator

                        elif optimize == 'True': # what to do if the user would like to run an optimization campaign
                                        
                                name = 'custom'
                                dataset = make_dataset(name)
                                opt(dataset,name,algorithm,str(goal),int(max_iter),int(campaign_number),emulator=False)
                                # run optimization campaign on dataset

                        email_address = str(request.form.get("email_address")) # users email address for results
                        msg = Message('Hey there', recipients=[email_address]) # subject line
                        msg.html = '<b> Thank you for using the Gomes Group DataShare. Attached is the data you requested!</b>' # a short message

                        if str(request.form.get("e_file")) == 'yes':

                                with app.open_resource("emulators/emulator_custom_BayesNeuralNet/emulator.pickle") as f:
                                        msg.attach("emultor.pickle", "application/python-pickle", f.read())

                        if str(request.form.get("o_file")) == 'yes':

                                with app.open_resource("campaigns/campaign_custom/opts.csv") as f:
                                        msg.attach("opts.csv", "text/csv", f.read())
                                
                        mail.send(msg)

                        try: # making a directory for the dataset (for initial campaigns)
                                os.mkdir("datasets/dataset_custom")
                        except: # remove config and description
                                try:
                                        os.remove("datasets/dataset_custom/config.json")
                                        os.remove("datasets/dataset_custom/description.txt")
                                except:
                                        pass
                                pass
        
                        return "Message sent!"

        return render_template("upload.html")
