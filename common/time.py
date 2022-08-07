from datetime import date, datetime

def today_():
    today = date.today()
    return today.strftime("%Y%m%d")

def now_(fmt=None):
    now = datetime.now()
    #return now.strftime("%Y%m%d.%H:%M:%S")
    if fmt is None:
        return now.strftime("%Y%m%d.%H_%M_%S")
    else:
        return now.strftime(fmt)
