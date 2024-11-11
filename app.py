from flask import Flask, render_template, request, redirect, url_for, flash, session
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import time
from urllib.parse import quote
import torch
from transformers import RobertaTokenizer
import pickle
from flask_mail import Mail, Message


# Initialize app and configure PostgreSQL
username = quote('postgres')
password = quote('dhruvi@268')
dbname = quote('mental_health_db')
app = Flask(__name__)
app.secret_key = 'your_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = f'postgresql://{username}:{password}@localhost/{dbname}'
db = SQLAlchemy(app)
TLS_CONFIG = {
    'MAIL_SERVER': 'smtp.gmail.com',
    'MAIL_PORT': 587,
    'MAIL_USE_TLS': True,
    'MAIL_USE_SSL': False
}

SSL_CONFIG = {
    'MAIL_SERVER': 'smtp.gmail.com',
    'MAIL_PORT': 465,
    'MAIL_USE_TLS': False,
    'MAIL_USE_SSL': True
}

# Common email configuration
app.config.update({
    'MAIL_USERNAME': 'hello.calmsea@gmail.com',
    'MAIL_PASSWORD': 'pzil qxxj ryjv xjlb',
      # Use app-specific password if 2FA is enabled
})

# Choose configuration mode
def configure_mail(mode='SSL'):
    if mode == 'TLS':
        app.config.update(TLS_CONFIG)
    else:
        app.config.update(SSL_CONFIG)

# Initialize mail with SSL configuration by default
configure_mail(mode='SSL')
mail = Mail(app)

# app.config['MAIL_SERVER'] = 'smtp.gmail.com'
# app.config['MAIL_PORT'] = 587
# app.config['MAIL_USE_TLS'] = True

# mail = Mail(app)

# Define User and Details models
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    firstname = db.Column(db.String(50), nullable=False)
    lastname = db.Column(db.String(50), nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    depression_score = db.Column(db.Integer, default=0)
    depression_state = db.Column(db.String(20), default="unknown")
    details_completed = db.Column(db.Boolean, default=False)

class Details(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    age = db.Column(db.Integer, nullable=False)
    gender = db.Column(db.String(10), nullable=False)
    occupation = db.Column(db.String(100), nullable=False)
    mobile = db.Column(db.String(15), nullable=False)

# Create database tables
with app.app_context():
    db.create_all()

# Load tokenizer for RoBERTa-based text classification
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

def load_model_from_pkl(filename):
    try:
        with open(filename, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# Load the model and label encoder
import os

# Define absolute paths
model_path = os.path.abspath('roberta_model3.pkl')
label_encoder_path = os.path.abspath('label_encoder.pkl')
model = load_model_from_pkl(model_path)
if model is None:
    print("Model could not be loaded. Check the path and ensure it is a valid pickle file.")

try:
    with open('label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
except Exception as e:
    print(f"Error loading label encoder: {e}")
    label_encoder = None

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
if model:
    model = model.to(device)
    model.eval()

# Route for the welcome page
@app.route('/')
def welcome():
    return render_template('welcome.html')

@app.route('/Aboutus')
def about():
    if 'user_id' not in session:
        return redirect(url_for('index'))
    return render_template('Aboutus.html')
    

@app.route('/contact')
def contact():
     return render_template('contact.html')

@app.route('/send_email',methods=['POST'])
def send_email():
    # if request.method == 'POST':
    #    mode=request.form.get('mode','SSL')
    #    configure_mail(mode)
       name=request.form['name']
       user_email=request.form['email']
       message=request.form['message']

       msg=Message(subject="New Contact for Submission",
                   sender=app.config['MAIL_USERNAME'],
                   recipients=["hello.calmsea@gmail.com"],
                   reply_to=user_email)
       msg.body=f"Name: {name}\nEmail: {user_email}\nMessage:\n{message}"

       try:
           mail.send(msg)
           flash("Message sent Successfully!","success")
       except Exception as e:
           flash(f"An error occurred: {e}","danger")
       
       return render_template('contact.html')

# Route for index page handling login and registration
def send_login_email(user):
    subject = "Welcome Back! You've Logged In"
    body = (
        f"Hello {user.firstname},\n\n"
        "Thank you for logging into our mental health detection website. "
        "We're here to support you every step of the way. "
        "Wishing you positivity and strength!\n\nBest Regards,\nThe Team"
    )
    msg = Message(subject=subject, sender=app.config['MAIL_USERNAME'], recipients=[user.email])
    msg.body = body
    mail.send(msg)
# Define login-required decorator
def login_required(route_func):
    def wrapper(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('index'))
        return route_func(*args, **kwargs)
    wrapper.__name__ = route_func.__name__
    return wrapper

# Route for index (login and registration)
@app.route('/index', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        action = request.form.get('action')
        email = request.form.get('email')
        password = request.form.get('password')

        if action == 'register':
            firstname = request.form.get('firstname')
            lastname = request.form.get('lastname')
            hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
            try:
                new_user = User(firstname=firstname, lastname=lastname, email=email, password=hashed_password)
                db.session.add(new_user)
                db.session.commit()
                flash('Registration successful!', 'success')
                return redirect(url_for('index'))
            except Exception as e:
                flash(f'Error: {str(e)}', 'danger')

        elif action == 'login':
            user = User.query.filter_by(email=email).first()
            if user and check_password_hash(user.password, password):
                session['user_id'] = user.id
                session['user_name'] = user.firstname
                send_login_email(user)  # Send login notification email
                if not user.details_completed:
                    flash('Please complete your details before proceeding.', 'info')
                    return redirect(url_for('details'))
                flash(f'Welcome, {user.firstname}!', 'success')
                return redirect(url_for('options'))
            else:
                flash('Invalid email or password.', 'danger')

    return render_template('index.html')

# Route for user options page
@app.route('/options')
@login_required
def options():
    if 'user_id' not in session:
        return redirect(url_for('index'))
    return render_template('option.html', user_name=session.get('user_name'))

# Route for user details page
@app.route('/details', methods=['GET', 'POST'])
@login_required
def details():
    if 'user_id' not in session:
        return redirect(url_for('index'))

    if request.method == 'POST':
        full_name = request.form.get('name').strip().split()
        firstname = full_name[0]
        lastname = ' '.join(full_name[1:]) if len(full_name) > 1 else ''
        age = request.form.get('age')
        gender = request.form.get('gender')
        occupation = request.form.get('occupation')
        mobile = request.form.get('mobile')

        user_id = session['user_id']
        new_details = Details(user_id=user_id, age=age, gender=gender, occupation=occupation, mobile=mobile)
        db.session.add(new_details)

        user = User.query.get(user_id)
        user.details_completed = True
        db.session.commit()

        flash('Details updated successfully.', 'success')
        return redirect(url_for('questionnaire'))

    return render_template('details.html')

# Route for questionnaire page
@app.route('/questionnaire', methods=['GET', 'POST'])
@login_required
def questionnaire():
    if 'user_id' not in session:
        return redirect(url_for('index'))

    user_name = session.get('user_name')
    return render_template('questionnaire.html', user_name=user_name)

# Route for results page with deep dive option
# Add the send_result_email function here (before the `results` route)
def send_result_email(user, depression_score, depression_status):
    recommendation_message = ""
    
    # Define recommendation messages based on depression score ranges
    if depression_score <= 5:
        recommendation_message = "You’re doing well! Keep up the positive habits and take care of yourself!"
    elif 6 <= depression_score <= 10:
        recommendation_message = "We recommend trying relaxation techniques and spending time with supportive people."
    elif 11 <= depression_score <= 15:
        recommendation_message = "Consider speaking with a mental health professional for guidance and support."
    else:
        recommendation_message = "It may be beneficial to seek a deeper assessment with a mental health expert."

    subject = "Your Depression Test Results and Recommendations"
    body = (
        f"Hello {user.firstname},\n\n"
        "Thank you for visiting our platform and trusting us with your mental health journey.\n\n"
        f"Here are your depression test results:\n\n"
        f"Depression Score: {depression_score}\n"
        f"Depression Status: {depression_status}\n\n"
        f"{recommendation_message}\n\n"
        "Remember, small steps make a big difference. Stay positive and reach out if you need support.\n\n"
        "Best regards,\nThe Mental Health Platform Team"
    )
    
    msg = Message(subject=subject, sender=app.config['MAIL_USERNAME'], recipients=[user.email])
    msg.body = body
    mail.send(msg)

# Modify the results route to include the email function
@app.route('/results', methods=['POST'])
@login_required
def results():
    scores = []
    for i in range(1, 14):
        score = request.form.get(f'q{i}', type=int)
        scores.append(score)

    total_score = sum(scores)

    if total_score <= 5:
        depression_status = "No depression"
        show_deeper_option = False
    elif 6 <= total_score <= 10:
        depression_status = "Mild depression"
        show_deeper_option = False
    elif 11 <= total_score <= 15:
        depression_status = "Moderate depression"
        show_deeper_option = True
    else:
        depression_status = "Severe depression"
        show_deeper_option = True

    # Update depression score and state in the database
    if 'user_id' in session:
        user = User.query.get(session['user_id'])
        user.depression_score = total_score
        user.depression_state = depression_status
        db.session.commit()
        
        # Send result email after updating user information
        send_result_email(user, total_score, depression_status)

    return render_template('results.html', score=total_score, status=depression_status, show_deeper_option=show_deeper_option)


# Route for deep dive analysis page
@app.route('/modelresult', methods=['GET', 'POST'])
@login_required
def deeper_analysis():
    if 'user_id' not in session:
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        text = request.form['text']

        if model is None:  # Ensure model is valid
            flash('Model is not loaded. Please try again later.', 'danger')
            return redirect(url_for('options'))

        predicted_class, confidence = custom_predict(text, model, tokenizer, device)
        return render_template('modelresult.html', text=text, predicted_class=predicted_class, confidence=confidence)

    return render_template('modelresult.html')

# Prediction function using RoBERTa model
def custom_predict(text, model, tokenizer, device):
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=128,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)
        max_prob, predicted_class = torch.max(probabilities, dim=1)
        predicted_label = label_encoder.inverse_transform([predicted_class.item()])[0]

    return predicted_label, max_prob.item()

# Logout Route
@app.route('/logout',methods=['POST'])
def logout():
    # session.pop('user_id', None)
    # session.pop('user_name', None)
    # flash('You have been logged out.', 'success')
    session.pop('user_id', None)
    session.clear()
    flash('You have been logged out.', 'success')
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
