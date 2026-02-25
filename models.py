from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()


class UserProfile(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    zipf_threshold = db.Column(db.Float, default=4.0)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class WordList(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    status = db.Column(db.String(20), default='processing')  # processing, done, error
    words = db.relationship('Word', backref='word_list', lazy=True, cascade='all, delete-orphan')


class Word(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    word_list_id = db.Column(db.Integer, db.ForeignKey('word_list.id'), nullable=False)
    word = db.Column(db.String(100), nullable=False)
    zipf_score = db.Column(db.Float, nullable=False)
    mastered = db.Column(db.Boolean, default=False)
    attempts = db.relationship('QuizAttempt', backref='word_entry', lazy=True, cascade='all, delete-orphan')


class QuizAttempt(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    word_id = db.Column(db.Integer, db.ForeignKey('word.id'), nullable=False)
    user_definition = db.Column(db.Text, nullable=False)
    score = db.Column(db.Integer, nullable=False)
    reason = db.Column(db.Text)
    official_definition = db.Column(db.Text)
    synonyms = db.Column(db.Text)
    example = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
