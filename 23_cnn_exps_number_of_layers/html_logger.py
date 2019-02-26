import datetime
import os as os

LOG_IMG_DIR = "logimgs"

class html_logger:

    # create a new logger instance
    # write html header and prepare html body
    def __init__(self, filename):

        self.next_image_nr = 0

        self.logfile = open(filename, "w")
        self.logfile.write("<html>\n")
        self.logfile.write("<head>\n")
        self.logfile.write("<title>\n")
        self.logfile.write("Logfile\n")
        self.logfile.write("</title>\n")
        self.logfile.write("</head>\n")
        self.logfile.write("<body>\n")

        if not os.path.exists(LOG_IMG_DIR):
            os.makedirs(LOG_IMG_DIR)


    # get current date and time as a string
    # so we can annotate each log message log
    # message with a time stamp
    def get_date_time_str(self):

        now = datetime.datetime.now()
        datetimestr = str(now.day).zfill(2)    + "."   +\
                      str(now.month).zfill(2)  + "."   +\
                      str(now.year).zfill(4)   + " - " +\
                      str(now.hour).zfill(2)   + ":"   +\
                      str(now.minute).zfill(2) + ":"   +\
                      str(now.second).zfill(2)
        return datetimestr


    # write a string into the log file
    def log_msg(self, msg):

        if msg=="":
            self.logfile.write("<br>\n")
        else:
            complete_msg = self.get_date_time_str() + " : " + msg

            # output message to html file
            self.logfile.write( complete_msg + "<br>\n")

            # output message to console
            #print( complete_msg )

        self.logfile.flush()


    # log a whole plot
    def log_pyplot(self, plt):

        img_filename = self.get_new_image_filename()
        plt.savefig(img_filename)
        self.log_img_by_file(img_filename)
        self.logfile.flush()


    # log an image that already exists as  file
    def log_img_by_file(self, filename):

        self.logfile.write('<img src="' + filename + '"><br><br>\n')
        self.logfile.flush()


    # get a new image log filename
    def get_new_image_filename(self):

        self.next_image_nr += 1
        new_image_filename = LOG_IMG_DIR + "\\img_" + str(self.next_image_nr).zfill(5) + ".png"
        return new_image_filename


    # finalize the html logfile
    # by writing the closing body and html tags
    def close(self):
        self.logfile.write("</body>")
        self.logfile.write("</html>")
