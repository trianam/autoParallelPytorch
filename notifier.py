import smtplib
from email.mime.text import MIMEText
from notifierConfig import account

def sendMessage(subject, message=''):
	msg = MIMEText(message)

	msg['Subject'] = subject
	msg['From'] = account.fromMail
	msg['To'] = account.toMail

	smtpObj = smtplib.SMTP_SSL(account.smtpServer, account.smtpPort)
	smtpObj.login(account.userName, account.password)
	smtpObj.send_message(msg)
	smtpObj.quit()

def sendFile(subject, fileName, message=''):
	with open(fileName) as fp:
		msg = MIMEText(message+"\n"+fp.read())

	msg['Subject'] = subject
	msg['From'] = account.fromMail
	msg['To'] = account.toMail

	smtpObj = smtplib.SMTP_SSL(account.smtpServer, account.smtpPort)
	smtpObj.login(account.userName, account.password)
	smtpObj.send_message(msg)
	smtpObj.quit()

