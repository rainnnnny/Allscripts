
# default args
port = 25 # SMTP protocol default
# from_addr = input('email account: ')
# smtp_server = 'smtp.%s' % from_addr.split('@')[1]

### qq
# from_addr = '413414606@qq.com'
# smtp_server = 'smtp.qq.com'
# port = 587
###

### sina
from_addr = 'rainnnnny@sina.com'
smtp_server = 'smtp.sina.com'
###

to_addr = 'rainnnnny@sina.com'
sMsg = 'you got it.'

import email
import smtplib


def send(msg=''):
    password = input('password: ') if not msg else input(msg)

    msg = email.message_from_string(sMsg)
    msg['From'] = 'nobody <%s>' % from_addr
    msg['To'] = 'hello <%s>' % to_addr
    msg['Subject'] = 'Greetings from the Big Brother'

    server = smtplib.SMTP(smtp_server, port)
    server.set_debuglevel(1)
    server.starttls()
    try:
        server.login(from_addr, password)
    except:
        send('\n\n\npassword error, try again:')
        return
    server.sendmail(from_addr, [to_addr], msg.as_string())
    server.quit()

send()