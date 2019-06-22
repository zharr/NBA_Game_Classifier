import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

def predict(df):
    #df_p = pd.read_csv('todays_games.csv')
    retDF = pd.DataFrame(columns=['Away', 'WP_A', 'Home', 'WP_H', 'Winner', 'ActualWinner', 'Date'])
    df_p = df
    df_predict = df_p[['FG%_A', 'FT%_A',
                       'FG%_H', 'FT%_H']]
    x_val = df_predict

    mean = x_val['FG%_A'].mean()
    std = x_val['FG%_A'].std()
    x_val.loc[:, 'FG%_A'] = (x_val['FG%_A'] - mean) / std
    mean = x_val['FT%_A'].mean()
    std = x_val['FT%_A'].std()
    x_val.loc[:, 'FT%_A'] = (x_val['FT%_A'] - mean) / std
    mean = x_val['FG%_H'].mean()
    std = x_val['FG%_H'].std()
    x_val.loc[:, 'FG%_H'] = (x_val['FG%_H'] - mean) / std
    mean = x_val['FT%_H'].mean()
    std = x_val['FT%_H'].std()
    x_val.loc[:, 'FT%_H'] = (x_val['FT%_H'] - mean) / std

    x_val = x_val.values

    # create array for keeping track of stochastic part
    sto = []
    for i in range(x_val.shape[0]):
        d = {}
        d[df_p.iloc[i,0]] = 0
        d[df_p.iloc[i,4]] = 0
        sto.append(d)

    runs = 10000

    with tf.Session() as sess:
        # Initialize variables
        #sess.run(tf.global_variables_initializer())

        # Restore model weights from previously saved model
        saver = tf.train.import_meta_graph('./tmp/model.ckpt.meta')
        saver.restore(sess, "./tmp/model.ckpt")
        #saver.restore(sess, './tmp/model.ckpt')

        # get pred_label and inputs back
        graph = tf.get_default_graph()
        inputs = graph.get_tensor_by_name("inputs:0")
        pred_label = graph.get_tensor_by_name("pred_label:0")

        for a in range(runs):
            for i in range(x_val.shape[0]):
                winner = sess.run(pred_label, feed_dict={inputs:x_val[i,None]})
                home =df_p.iloc[i,4]
                away = df_p.iloc[i,0]
                if winner == 0:
                    #print(away + " wins vs. " + home)
                    sto[i][away] += 1
                else:
                    #print(home + " wins vs. " + away)
                    sto[i][home] += 1
        sess.close()

    for i in range(len(sto)):
        d = sto[i]
        teams = []
        chance = []
        for key, val in d.items():
            teams.append(key)
            chance.append(val)
        print(str(teams[0])+': '+"{0:.2f}".format((chance[0]/runs)*100)+"%  |  "+teams[1]+': '+"{0:.2f}".format((chance[1]/runs)*100)+"%")
        print("--------------------------------------------------------------")
        if ((chance[0]/runs)*100) > ((chance[1]/runs)*100):
            winner = 0
        else:
            winner = 1
        tmpDF = pd.DataFrame(data=[[teams[0],((chance[0]/runs)*100),teams[1],((chance[1]/runs)*100),winner,-1,pd.to_datetime('today').strftime('%Y-%m-%d')]], columns=['Away', 'WP_A', 'Home', 'WP_H', 'Winner', 'ActualWinner', 'Date'])
        retDF = retDF.append(tmpDF, ignore_index=True)
    return retDF

def summarize(df):
    print(df.head(30))
    tmpDF = df.filter(['Away','WP_A','Home','WP_H'], axis=1)
    tmpDF.to_csv('todays_picks.csv', index=False)
    with open('all_picks.csv', 'a') as f:
        df.to_csv(f, header=False, index=False)
    #send_message()
    return df

def send_message(emailto):
    import smtplib
    import mimetypes
    from email.mime.multipart import MIMEMultipart
    from email import encoders
    from email.message import Message
    from email.mime.audio import MIMEAudio
    from email.mime.base import MIMEBase
    from email.mime.image import MIMEImage
    from email.mime.text import MIMEText

    fileToSend = "todays_picks.csv"
    subject = "Today's NBA Picks"
    body = "Attached are the picks."
    emailfrom = "nba.modeler@gmail.com"
    #emailto = "zach.harrison55@gmail.com"
    password = 'iktsuarpok5'

    msg = MIMEMultipart()
    msg["From"] = emailfrom
    msg["To"] = emailto
    msg["Subject"] = subject
    msg.preamble = "help I cannot send an attachment to save my life"

    ctype, encoding = mimetypes.guess_type(fileToSend)
    if ctype is None or encoding is not None:
        ctype = "application/octet-stream"

    maintype, subtype = ctype.split("/", 1)

    if maintype == "text":
        fp = open(fileToSend)
        # Note: we should handle calculating the charset
        attachment = MIMEText(fp.read(), _subtype=subtype)
        fp.close()
    elif maintype == "image":
        fp = open(fileToSend, "rb")
        attachment = MIMEImage(fp.read(), _subtype=subtype)
        fp.close()
    elif maintype == "audio":
        fp = open(fileToSend, "rb")
        attachment = MIMEAudio(fp.read(), _subtype=subtype)
        fp.close()
    else:
        fp = open(fileToSend, "rb")
        attachment = MIMEBase(maintype, subtype)
        attachment.set_payload(fp.read())
        fp.close()
        encoders.encode_base64(attachment)
    attachment.add_header("Content-Disposition", "attachment", filename=fileToSend)
    msg.attach(attachment)

    server = smtplib.SMTP("smtp.gmail.com:587")
    server.starttls()
    server.login(emailfrom, password)
    server.sendmail(emailfrom, emailto, msg.as_string())
    server.quit()


def send_message_old():
    import email, smtplib, ssl

    from email import encoders
    from email.mime.base import MIMEBase
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText

    subject = "Today's NBA Picks"
    body = "Attached are the picks."
    sender_email = "nba.modeler@gmail.com"
    receiver_email = "zach.harrison55@gmail.com"
    password = 'iktsuarpok5'

    # Create a multipart message and set headers
    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = receiver_email
    message["Subject"] = subject
    #message["Bcc"] = receiver_email  # Recommended for mass emails

    # Add body to email
    message.attach(MIMEText(body, "plain"))

    filename = "todays_picks.csv"  # In same directory as script

    # Open PDF file in binary mode
    with open(filename, "rb") as attachment:
        # Add file as application/octet-stream
        # Email client can usually download this automatically as attachment
        part = MIMEBase("application", "octet-stream")
        part.set_payload(attachment.read())


    # Encode file in ASCII characters to send by email
    encoders.encode_base64(part)

    # Add attachment to message and convert message to string
    message.attach(part)
    text = message.as_string()

    # Log in to server using secure context and send email
    context = ssl.create_default_context()
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, text)
