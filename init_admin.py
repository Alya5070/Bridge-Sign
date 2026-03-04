import os
import sys
import logging

from app2 import app, db, User

with app.app_context():
    db.create_all()
    admin_user = User.query.filter_by(username='Admin').first()
    if not admin_user:
        admin_user = User(username='Admin', is_admin=True)
        admin_user.set_password('12356')
        db.session.add(admin_user)
        db.session.commit()
        print("Created Admin user")
    elif not admin_user.is_admin:
        admin_user.is_admin = True
        admin_user.set_password('12356')
        db.session.commit()
        print("Updated Admin user")
    else:
        admin_user.set_password('12356')
        db.session.commit()
        print("Admin user already has admin privileges, reset password")
