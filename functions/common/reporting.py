# Copyright (C) 2019 Christopher Gearhart
# chris@bblanimation.com
# http://bblanimation.com/
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# System imports
import sys
import os
import platform
import traceback
import math
import time

# Blender imports
import bpy

# Module imports
# NONE!


def stopwatch(text:str, start_time:float, end_time:float=None, precision:int=5, multiplier:float=1):
    """ from seconds to Days;Hours:Minutes;Seconds """
    end_time = end_time or time.time()
    value = end_time - start_time
    value = value * multiplier

    d_value = (((value / 365) / 24) / 60)
    days = int(d_value)

    h_value = (d_value - days) * 365
    hours = int(h_value)

    m_value = (h_value - hours) * 24
    minutes = int(m_value)

    s_value = (m_value - minutes) * 60
    seconds = round(s_value, precision)

    output_string = str(text) + ": " + str(days) + ";" + str(hours) + ":" + str(minutes) + ";" + str(seconds)
    print(output_string)
    return time.time()  # TIP: store this to variable and call 'stopwatch' again later with this as start time


def update_progress_bars(print_status:bool, cursor_status:bool, cur_percent:float, old_percent:float, status_type:str, end:bool=False):
    """ print updated progress bar and update progress cursor """
    if print_status and cur_percent - old_percent > 0.001 and (cur_percent < 1 or end):
        update_progress(status_type, cur_percent)
        if cursor_status and math.ceil(cur_percent * 100) != math.ceil(old_percent * 100):
            wm = bpy.context.window_manager
            if cur_percent == 0:
                wm.progress_begin(0, 100)
            elif cur_percent < 1:
                wm.progress_update(math.floor(cur_percent*100))
            else:
                wm.progress_end()
        old_percent = cur_percent
    return old_percent


def update_progress(job_title:str, progress:float):
    """ print updated progress bar """
    length = 20  # modify this to change the length
    block = int(round(length * progress))
    msg = "\r{0}: [{1}] {2}%".format(job_title, ("#" * block) + ("-" * (length - block)), round(progress * 100, 1))
    if progress >= 1:
        msg += " DONE\r\n"
    sys.stdout.write(msg)
    sys.stdout.flush()


# https://github.com/CGCookie/retopoflow
def show_error_message(message:str, wrap:int=80):
    if not message or wrap == 0:
        return
    lines = message.splitlines()
    nlines = []
    for line in lines:
        spc = len(line) - len(line.lstrip())
        while len(line) > wrap:
            i = line.rfind(" ", 0, wrap)
            if i == -1:
                nlines += [line[:wrap]]
                line = line[wrap:]
            else:
                nlines += [line[:i]]
                line = line[i+1:]
            if line:
                line = " "*spc + line
        nlines += [line]
    lines = nlines

    def draw(self,context):
        for line in lines:
            self.layout.label(text=line)

    bpy.context.window_manager.popup_menu(draw, title="Error Message", icon="ERROR")
    return


def handle_exception(log_name:str, report_button_loc:str):
    errormsg = print_exception(log_name)
    # if max number of exceptions occur within threshold of time, abort!
    error_str = "Something went wrong. Please start an error report with us so we can fix it! ('%(report_button_loc)s')" % locals()
    print("\n"*5)
    print("-"*100)
    print(error_str)
    print("-"*100)
    print("\n"*5)
    show_error_message(error_str, wrap=240)


def get_exception_message():
    exc_type, exc_obj, tb = sys.exc_info()

    errormsg = "EXCEPTION (%s): %s\n" % (exc_type, exc_obj)
    etb = traceback.extract_tb(tb)
    pfilename = None
    for i, entry in enumerate(reversed(etb)):
        filename, lineno, funcname, line = entry
        if filename != pfilename:
            pfilename = filename
            errormsg += "         %s\n" % (filename)
        errormsg += "%03d %04d:%s() %s\n" % (i, lineno, funcname, line.strip())

    return errormsg


# http://stackoverflow.com/questions/14519177/python-exception-handling-line-number
def print_exception(text_name:str, show_error:bool=False, errormsg:str=""):
    errormsg = errormsg or get_exception_message()

    print(errormsg)

    # create a log file for error writing
    txt = bpy.data.texts.get(text_name)
    if txt is None:
        txt = bpy.data.texts.new(text_name)
    else:
        txt.clear()

    # write error to log text object
    txt.write(errormsg + "\n")

    if show_error:
        show_error_message(errormsg, wrap=240)

    return errormsg


# https://github.com/CGCookie/retopoflow
def bversion(short:bool=True):
    """ return Blender version string """
    major,minor,rev = bpy.app.version
    bver_long = "%03d.%03d.%03d" % (major,minor,rev)
    bver_short = "%d.%02d" % (major, minor)
    return bver_short if short else bver_long


def b280():
    return bpy.app.version >= (2,80,0)


def write_error_to_file(error_report_path:str, error_log:str, addon_version:str, github_path:str):
    # write error to log text object
    error_report_dir = os.path.dirname(error_report_path)
    if not os.path.exists(error_report_dir):
        os.makedirs(error_report_dir)
    f = open(error_report_path, "w")
    f.write("\nPlease copy the following form and paste it into a new issue at " + github_path)
    f.write("\n\nDon't forget to include a description of your problem! The more information you provide (what you were trying to do, what action directly preceeded the error, etc.), the easier it will be for us to squash the bug.")
    f.write("\n\n### COPY EVERYTHING BELOW THIS LINE ###\n")
    f.write("\nDescription of the Problem:\n")
    f.write("\nBlender Version: " + bversion(short=False))
    f.write("\nAddon Version: " + addon_version)
    f.write("\nPlatform Info:")
    f.write("\n   system   = " + platform.system())
    f.write("\n   platform = " + platform.platform())
    f.write("\n   version  = " + platform.version())
    f.write("\n   python   = " + platform.python_version())
    f.write("\nError:")
    try:
        f.write("\n" + error_log)
    except KeyError:
        f.write(" No exception found")
    f.close()
