import csv
from email.parser import BytesParser
from email.policy import default

# Replace 'your_eml_file.eml' with the actual path to your EML file
eml_file = "file_1.eml"

try:
    with open(eml_file, 'rb') as eml_file:
        # Parse the EML file using the email library
        msg = BytesParser(policy=default).parse(eml_file)

        # Access the email headers
        sender = msg['from']
        recipient = msg['to']
        subject = msg['subject']
        date = msg['date']

        # Extract email content (text and HTML)
        text_content = ""
        html_content = ""

        for part in msg.walk():
            content_type = part.get_content_type()
            if content_type == 'text/plain':
                text_content = part.get_payload(decode=True).decode()
            elif content_type == 'text/html':
                html_content = part.get_payload(decode=True).decode()

        # Create a list with the extracted data
        email_data = [sender, recipient, subject, date, text_content, html_content]

        # Define the CSV file name
        csv_file = 'email_data.csv'

        # Write the data to the CSV file
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Sender', 'Recipient', 'Subject', 'Date', 'Text Content', 'HTML Content'])
            writer.writerow(email_data)

        print(f"Email data saved to {csv_file}")
except FileNotFoundError:
    print("The file does not exist")
except Exception as e:
    print(f"An error occurred: {e}")
