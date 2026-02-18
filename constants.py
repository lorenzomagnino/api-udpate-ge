import pandas as pd

future_calendar_eua = pd.DataFrame(
    [
        {"Date": "03-Mar-25", "Volume": 3245500},
        {"Date": "04-Mar-25", "Volume": 3245500},
        {"Date": "05-Mar-25", "Volume": 2072500},
        {"Date": "06-Mar-25", "Volume": 3245500},
        {"Date": "07-Mar-25", "Volume": 1607000},
        {"Date": "10-Mar-25", "Volume": 3245500},
        {"Date": "11-Mar-25", "Volume": 3245500},
        {"Date": "13-Mar-25", "Volume": 3245500},
        {"Date": "14-Mar-25", "Volume": 1607000},
        {"Date": "17-Mar-25", "Volume": 3245500},
        {"Date": "18-Mar-25", "Volume": 3245500},
        {"Date": "19-Mar-25", "Volume": 2072500},
        {"Date": "20-Mar-25", "Volume": 3245500},
        {"Date": "21-Mar-25", "Volume": 1607000},
        {"Date": "24-Mar-25", "Volume": 3245500},
        {"Date": "25-Mar-25", "Volume": 3245500},
        {"Date": "27-Mar-25", "Volume": 3245500},
        {"Date": "28-Mar-25", "Volume": 1607000},
        {"Date": "31-Mar-25", "Volume": 3246500},
        # ----- April 2025 -----
        # Week 14 (spanning end of March into early April)
        {"Date": "01-Apr-25", "Volume": 3245500},  # Tuesday
        {"Date": "02-Apr-25", "Volume": 2072500},  # Wednesday
        {"Date": "03-Apr-25", "Volume": 3245500},  # Thursday
        {"Date": "04-Apr-25", "Volume": 1607000},  # Friday
        # Week 15
        {"Date": "07-Apr-25", "Volume": 3245500},  # Monday
        {"Date": "08-Apr-25", "Volume": 3245500},  # Tuesday
        {"Date": "10-Apr-25", "Volume": 3245500},  # Thursday
        {"Date": "11-Apr-25", "Volume": 1607000},  # Friday
        # Week 16
        {"Date": "14-Apr-25", "Volume": 3245500},  # Monday
        {"Date": "15-Apr-25", "Volume": 3245500},  # Tuesday
        {"Date": "16-Apr-25", "Volume": 2072500},  # Wednesday
        {"Date": "17-Apr-25", "Volume": 3245500},  # Thursday
        {"Date": "18-Apr-25", "Volume": 1607000},  # Friday
        # Week 17
        {"Date": "21-Apr-25", "Volume": 3245500},  # Monday (assumed pattern)
        {"Date": "22-Apr-25", "Volume": 3245500},  # Tuesday
        {"Date": "24-Apr-25", "Volume": 3245500},  # Thursday
        {"Date": "25-Apr-25", "Volume": 1607000},  # Friday
        # ----- May 2025 -----
        # Week 19
        {"Date": "05-May-25", "Volume": 3245500},  # Monday
        {"Date": "06-May-25", "Volume": 3245500},  # Tuesday
        {"Date": "08-May-25", "Volume": 3245500},  # Thursday
        {"Date": "09-May-25", "Volume": 1607000},  # Friday
        # Week 20
        {"Date": "12-May-25", "Volume": 3245500},  # Monday
        {"Date": "13-May-25", "Volume": 3245500},  # Tuesday
        {"Date": "14-May-25", "Volume": 2072500},  # Wednesday
        {"Date": "15-May-25", "Volume": 3245500},  # Thursday
        {"Date": "16-May-25", "Volume": 1607000},  # Friday
        # Week 21
        {"Date": "19-May-25", "Volume": 3245500},  # Monday
        {"Date": "20-May-25", "Volume": 3245500},  # Tuesday
        {"Date": "22-May-25", "Volume": 3245500},  # Thursday
        {"Date": "23-May-25", "Volume": 1607000},  # Friday
        # Week 22
        {"Date": "26-May-25", "Volume": 3245500},  # Monday
        {"Date": "27-May-25", "Volume": 3245500},  # Tuesday
        {"Date": "28-May-25", "Volume": 2072500},  # Wednesday
        # ----- June 2025 -----
        # Week 23
        {"Date": "02-Jun-25", "Volume": 3245500},  # Monday
        {"Date": "03-Jun-25", "Volume": 3245500},  # Tuesday
        {"Date": "05-Jun-25", "Volume": 3245500},  # Thursday
        {"Date": "06-Jun-25", "Volume": 1607000},  # Friday
        # Week 24
        {"Date": "09-Jun-25", "Volume": 3245500},  # Monday
        {"Date": "10-Jun-25", "Volume": 3245500},  # Tuesday
        {"Date": "11-Jun-25", "Volume": 2072500},  # Wednesday
        {"Date": "12-Jun-25", "Volume": 3245500},  # Thursday
        {"Date": "13-Jun-25", "Volume": 1607000},  # Friday
        # Week 25
        {"Date": "16-Jun-25", "Volume": 3245500},  # Monday
        {"Date": "17-Jun-25", "Volume": 3245500},  # Tuesday
        {"Date": "19-Jun-25", "Volume": 3245500},  # Thursday
        {"Date": "20-Jun-25", "Volume": 1607000},  # Friday
        # Week 26
        {"Date": "23-Jun-25", "Volume": 3245500},  # Monday
        {"Date": "24-Jun-25", "Volume": 3245500},  # Tuesday
        {"Date": "25-Jun-25", "Volume": 2072500},  # Wednesday
        {"Date": "26-Jun-25", "Volume": 3245500},  # Thursday
        {"Date": "27-Jun-25", "Volume": 1607000},  # Friday
        # ----- July 2025 -----
        # Week 27
        {"Date": "30-Jun-25", "Volume": 3246500},  # Monday (slight variation)
        {"Date": "01-Jul-25", "Volume": 3245500},  # Tuesday
        {"Date": "03-Jul-25", "Volume": 3245500},  # Thursday
        {"Date": "04-Jul-25", "Volume": 1607000},  # Friday
        # Week 28
        {"Date": "07-Jul-25", "Volume": 3245500},  # Monday
        {"Date": "08-Jul-25", "Volume": 3245500},  # Tuesday
        {"Date": "09-Jul-25", "Volume": 2072500},  # Wednesday
        {"Date": "10-Jul-25", "Volume": 3245500},  # Thursday
        {"Date": "11-Jul-25", "Volume": 1607000},  # Friday
        # Week 29
        {"Date": "14-Jul-25", "Volume": 3245500},  # Monday
        {"Date": "15-Jul-25", "Volume": 3245500},  # Tuesday
        {"Date": "17-Jul-25", "Volume": 3245500},  # Thursday
        {"Date": "18-Jul-25", "Volume": 1607000},  # Friday
        # Week 30
        {"Date": "21-Jul-25", "Volume": 3245500},  # Monday
        {"Date": "22-Jul-25", "Volume": 3245500},  # Tuesday
        {"Date": "23-Jul-25", "Volume": 2072500},  # Wednesday
        {"Date": "24-Jul-25", "Volume": 3245500},  # Thursday
        {"Date": "25-Jul-25", "Volume": 1607000},  # Friday
        # Week 31
        {"Date": "28-Jul-25", "Volume": 3245500},  # Monday
        {"Date": "29-Jul-25", "Volume": 3245500},  # Tuesday
        {"Date": "31-Jul-25", "Volume": 3245500},  # Thursday
        # ----- August 2025 -----
        # Note: Week 31's Friday falls on 01-Aug-25
        {"Date": "01-Aug-25", "Volume": 1607000},  # Friday from Week 31
        # Week 32
        {"Date": "04-Aug-25", "Volume": 3245500},  # Monday
        {"Date": "05-Aug-25", "Volume": 3245500},  # Tuesday
        {"Date": "06-Aug-25", "Volume": 2072500},  # Wednesday
        {"Date": "07-Aug-25", "Volume": 3245500},  # Thursday
        {"Date": "08-Aug-25", "Volume": 1607000},  # Friday
        # Week 33
        {"Date": "11-Aug-25", "Volume": 3245500},  # Monday
        {"Date": "12-Aug-25", "Volume": 3245500},  # Tuesday
        {"Date": "14-Aug-25", "Volume": 3245500},  # Thursday
        {"Date": "15-Aug-25", "Volume": 1607000},  # Friday
        # Week 34
        {"Date": "18-Aug-25", "Volume": 3245500},  # Monday
        {"Date": "19-Aug-25", "Volume": 3245500},  # Tuesday
        {"Date": "20-Aug-25", "Volume": 2069000},  # Wednesday (slight variation)
        {"Date": "21-Aug-25", "Volume": 3245500},  # Thursday
        {"Date": "22-Aug-25", "Volume": 1607000},  # Friday
        # Week 35
        {"Date": "25-Aug-25", "Volume": 3245500},  # Monday
        {"Date": "26-Aug-25", "Volume": 3245500},  # Tuesday
        {"Date": "28-Aug-25", "Volume": 3244500},  # Thursday (variation)
        {"Date": "29-Aug-25", "Volume": 1614500},  # Friday (variation)
        # ----- September 2025 -----
        # Week 36 - Revised volumes per MSR adjustments
        {"Date": "01-Sep-25", "Volume": 3268000},  # Monday - CAP3
        {"Date": "02-Sep-25", "Volume": 3268000},  # Tuesday - CAP3
        {"Date": "03-Sep-25", "Volume": 2162500},  # Wednesday - Poland
        {"Date": "04-Sep-25", "Volume": 3268000},  # Thursday - CAP3
        {"Date": "05-Sep-25", "Volume": 1691000},  # Friday - Germany
        # Week 37 (Note: Wednesday is not listed)
        {"Date": "08-Sep-25", "Volume": 3268000},  # Monday - CAP3
        {"Date": "09-Sep-25", "Volume": 3268000},  # Tuesday - CAP3
        {"Date": "11-Sep-25", "Volume": 3268000},  # Thursday - CAP3
        {"Date": "12-Sep-25", "Volume": 1691000},  # Friday - Germany
        # Week 38
        {"Date": "15-Sep-25", "Volume": 3268000},  # Monday - CAP3
        {"Date": "16-Sep-25", "Volume": 3268000},  # Tuesday - CAP3
        {"Date": "17-Sep-25", "Volume": 2162500},  # Wednesday - Poland
        {"Date": "18-Sep-25", "Volume": 3268000},  # Thursday - CAP3
        {"Date": "19-Sep-25", "Volume": 1691000},  # Friday - Germany
        # Week 39 (Wednesday omitted)
        {"Date": "22-Sep-25", "Volume": 3268000},  # Monday - CAP3
        {"Date": "23-Sep-25", "Volume": 3268000},  # Tuesday - CAP3
        {"Date": "25-Sep-25", "Volume": 3268000},  # Thursday - CAP3
        {"Date": "26-Sep-25", "Volume": 1691000},  # Friday - Germany
        # ----- Transition into October (Week 40) -----
        {"Date": "29-Sep-25", "Volume": 3268000},  # Monday - CAP3
        {"Date": "30-Sep-25", "Volume": 3268000},  # Tuesday - CAP3
        # Week 40 continues in October
        {"Date": "01-Oct-25", "Volume": 2162500},  # Wednesday - Poland
        {"Date": "02-Oct-25", "Volume": 3268000},  # Thursday - CAP3
        {"Date": "03-Oct-25", "Volume": 1691000},  # Friday - Germany
        # ----- October 2025 -----
        # Week 41
        {"Date": "06-Oct-25", "Volume": 3268000},  # Monday - CAP3
        {"Date": "07-Oct-25", "Volume": 3268000},  # Tuesday - CAP3
        {
            "Date": "08-Oct-25",
            "Volume": 1047500,
        },  # special day (lower volume - Northern Ireland)
        {"Date": "09-Oct-25", "Volume": 3268000},  # Thursday - CAP3
        {"Date": "10-Oct-25", "Volume": 1691000},  # Friday - Germany
        # Week 42
        {"Date": "13-Oct-25", "Volume": 3268000},  # Monday - CAP3
        {"Date": "14-Oct-25", "Volume": 3268000},  # Tuesday - CAP3
        {"Date": "15-Oct-25", "Volume": 2162500},  # Wednesday - Poland
        {"Date": "16-Oct-25", "Volume": 3268000},  # Thursday - CAP3
        {"Date": "17-Oct-25", "Volume": 1691000},  # Friday - Germany
        # Week 43 (Wednesday omitted)
        {"Date": "20-Oct-25", "Volume": 3268000},  # Monday - CAP3
        {"Date": "21-Oct-25", "Volume": 3268000},  # Tuesday - CAP3
        {"Date": "23-Oct-25", "Volume": 3268000},  # Thursday - CAP3
        {"Date": "24-Oct-25", "Volume": 1691000},  # Friday - Germany
        # Week 44
        {"Date": "27-Oct-25", "Volume": 3268000},  # Monday - CAP3
        {"Date": "28-Oct-25", "Volume": 3268000},  # Tuesday - CAP3
        {"Date": "29-Oct-25", "Volume": 2162500},  # Wednesday - Poland
        {"Date": "30-Oct-25", "Volume": 3268000},  # Thursday - CAP3
        {"Date": "31-Oct-25", "Volume": 1691000},  # Friday - Germany
        # ----- November 2025 -----
        # Week 45 (Wednesday omitted)
        {"Date": "03-Nov-25", "Volume": 3268000},  # Monday - CAP3
        {"Date": "04-Nov-25", "Volume": 3268000},  # Tuesday - CAP3
        {"Date": "06-Nov-25", "Volume": 3268000},  # Thursday - CAP3
        {"Date": "07-Nov-25", "Volume": 1691000},  # Friday - Germany
        # Week 46
        {"Date": "10-Nov-25", "Volume": 3268000},  # Monday - CAP3
        {"Date": "11-Nov-25", "Volume": 3268000},  # Tuesday - CAP3
        {"Date": "12-Nov-25", "Volume": 2162500},  # Wednesday - Poland
        {"Date": "13-Nov-25", "Volume": 3268000},  # Thursday - CAP3
        {"Date": "14-Nov-25", "Volume": 1691000},  # Friday - Germany
        # Week 47 (Wednesday omitted)
        {"Date": "17-Nov-25", "Volume": 3268000},  # Monday - CAP3
        {"Date": "18-Nov-25", "Volume": 3268000},  # Tuesday - CAP3
        {"Date": "20-Nov-25", "Volume": 3268000},  # Thursday - CAP3
        {"Date": "21-Nov-25", "Volume": 1691000},  # Friday - Germany
        # Week 48
        {"Date": "24-Nov-25", "Volume": 3268000},  # Monday - CAP3
        {"Date": "25-Nov-25", "Volume": 3268000},  # Tuesday - CAP3
        {"Date": "26-Nov-25", "Volume": 2162500},  # Wednesday - Poland
        {"Date": "27-Nov-25", "Volume": 3268000},  # Thursday - CAP3
        {"Date": "28-Nov-25", "Volume": 1691000},  # Friday - Germany
        # ----- December 2025 -----
        # Week 49 (Wednesday omitted)
        {"Date": "01-Dec-25", "Volume": 3268000},  # Monday - CAP3
        {"Date": "02-Dec-25", "Volume": 3268000},  # Tuesday - CAP3
        {"Date": "04-Dec-25", "Volume": 3268000},  # Thursday - CAP3
        {"Date": "05-Dec-25", "Volume": 1691000},  # Friday - Germany
        # Week 50 (final auctions with slightly higher volumes)
        {"Date": "08-Dec-25", "Volume": 3268000},  # Monday - CAP3
        {"Date": "09-Dec-25", "Volume": 3268000},  # Tuesday - CAP3
        {"Date": "10-Dec-25", "Volume": 2166000},  # Wednesday - Poland (final auction)
        {"Date": "11-Dec-25", "Volume": 3273000},  # Thursday - CAP3 (final auction)
        {"Date": "12-Dec-25", "Volume": 1696000},  # Friday - Germany (final auction)
        # ----- January 2026 -----
        # Week 1 - Jan-Aug volumes: CAP3=2,712,500, DE=1,093,000, PL=1,524,500
        {"Date": "05-Jan-26", "Volume": 2712500},  # Monday - CAP3
        {"Date": "06-Jan-26", "Volume": 2712500},  # Tuesday - CAP3
        {"Date": "07-Jan-26", "Volume": 1524500},  # Wednesday - Poland
        {"Date": "08-Jan-26", "Volume": 2712500},  # Thursday - CAP3
        {"Date": "09-Jan-26", "Volume": 1093000},  # Friday - Germany
        # Week 2
        {"Date": "12-Jan-26", "Volume": 2712500},  # Monday - CAP3
        {"Date": "13-Jan-26", "Volume": 2712500},  # Tuesday - CAP3
        {"Date": "14-Jan-26", "Volume": 2712500},  # Thursday - CAP3
        {"Date": "15-Jan-26", "Volume": 1093000},  # Friday - Germany
        # Week 3
        {"Date": "19-Jan-26", "Volume": 2712500},  # Monday - CAP3
        {"Date": "20-Jan-26", "Volume": 2712500},  # Tuesday - CAP3
        {"Date": "21-Jan-26", "Volume": 1524500},  # Wednesday - Poland
        {"Date": "22-Jan-26", "Volume": 2712500},  # Thursday - CAP3
        {"Date": "23-Jan-26", "Volume": 1093000},  # Friday - Germany
        # Week 4
        {"Date": "26-Jan-26", "Volume": 2712500},  # Monday - CAP3
        {"Date": "27-Jan-26", "Volume": 2712500},  # Tuesday - CAP3
        {"Date": "28-Jan-26", "Volume": 2712500},  # Thursday - CAP3
        {"Date": "29-Jan-26", "Volume": 1093000},  # Friday - Germany
        # ----- February 2026 -----
        # Week 5
        {"Date": "02-Feb-26", "Volume": 2712500},  # Monday - CAP3
        {"Date": "03-Feb-26", "Volume": 2712500},  # Tuesday - CAP3
        {"Date": "04-Feb-26", "Volume": 1524500},  # Wednesday - Poland
        {"Date": "05-Feb-26", "Volume": 2712500},  # Thursday - CAP3
        {"Date": "06-Feb-26", "Volume": 1093000},  # Friday - Germany
        # Week 6
        {"Date": "09-Feb-26", "Volume": 2712500},  # Monday - CAP3
        {"Date": "10-Feb-26", "Volume": 2712500},  # Tuesday - CAP3
        {"Date": "11-Feb-26", "Volume": 2712500},  # Thursday - CAP3
        {"Date": "12-Feb-26", "Volume": 1093000},  # Friday - Germany
        # Week 7
        {"Date": "16-Feb-26", "Volume": 2712500},  # Monday - CAP3
        {"Date": "17-Feb-26", "Volume": 2712500},  # Tuesday - CAP3
        {"Date": "18-Feb-26", "Volume": 1524500},  # Wednesday - Poland
        {"Date": "19-Feb-26", "Volume": 2712500},  # Thursday - CAP3
        {"Date": "20-Feb-26", "Volume": 1093000},  # Friday - Germany
        # Week 8
        {"Date": "23-Feb-26", "Volume": 2712500},  # Monday - CAP3
        {"Date": "24-Feb-26", "Volume": 2712500},  # Tuesday - CAP3
        {"Date": "25-Feb-26", "Volume": 2712500},  # Thursday - CAP3
        {"Date": "26-Feb-26", "Volume": 1093000},  # Friday - Germany
        # ----- March 2026 -----
        # Week 9
        {"Date": "02-Mar-26", "Volume": 2712500},  # Monday - CAP3
        {"Date": "03-Mar-26", "Volume": 2712500},  # Tuesday - CAP3
        {"Date": "04-Mar-26", "Volume": 1524500},  # Wednesday - Poland
        {"Date": "05-Mar-26", "Volume": 2712500},  # Thursday - CAP3
        {"Date": "06-Mar-26", "Volume": 1093000},  # Friday - Germany
        # Week 10
        {"Date": "09-Mar-26", "Volume": 2712500},  # Monday - CAP3
        {"Date": "10-Mar-26", "Volume": 2712500},  # Tuesday - CAP3
        {"Date": "11-Mar-26", "Volume": 2712500},  # Thursday - CAP3
        {"Date": "12-Mar-26", "Volume": 1093000},  # Friday - Germany
        # Week 11
        {"Date": "16-Mar-26", "Volume": 2712500},  # Monday - CAP3
        {"Date": "17-Mar-26", "Volume": 2712500},  # Tuesday - CAP3
        {"Date": "18-Mar-26", "Volume": 1524500},  # Wednesday - Poland
        {"Date": "19-Mar-26", "Volume": 2712500},  # Thursday - CAP3
        {"Date": "20-Mar-26", "Volume": 1093000},  # Friday - Germany
        # Week 12
        {"Date": "23-Mar-26", "Volume": 2712500},  # Monday - CAP3
        {"Date": "24-Mar-26", "Volume": 2712500},  # Tuesday - CAP3
        {"Date": "25-Mar-26", "Volume": 2712500},  # Thursday - CAP3
        {"Date": "26-Mar-26", "Volume": 1093000},  # Friday - Germany
        # Week 13
        {"Date": "30-Mar-26", "Volume": 2712500},  # Monday - CAP3
        {"Date": "31-Mar-26", "Volume": 2712500},  # Tuesday - CAP3
        # ----- April 2026 -----
        # Week 13 continues
        {"Date": "01-Apr-26", "Volume": 1524500},  # Wednesday - Poland
        {"Date": "02-Apr-26", "Volume": 2712500},  # Thursday - CAP3
        {"Date": "03-Apr-26", "Volume": 1093000},  # Friday - Germany
        # Week 14
        {"Date": "06-Apr-26", "Volume": 2712500},  # Monday - CAP3
        {"Date": "07-Apr-26", "Volume": 2712500},  # Tuesday - CAP3
        {"Date": "09-Apr-26", "Volume": 2712500},  # Thursday - CAP3
        {"Date": "10-Apr-26", "Volume": 1093000},  # Friday - Germany
        # Week 15
        {"Date": "13-Apr-26", "Volume": 2712500},  # Monday - CAP3
        {"Date": "14-Apr-26", "Volume": 2712500},  # Tuesday - CAP3
        {"Date": "15-Apr-26", "Volume": 1524500},  # Wednesday - Poland
        {"Date": "16-Apr-26", "Volume": 2712500},  # Thursday - CAP3
        {"Date": "17-Apr-26", "Volume": 1093000},  # Friday - Germany
        # Week 16
        {"Date": "20-Apr-26", "Volume": 2712500},  # Monday - CAP3
        {"Date": "21-Apr-26", "Volume": 2712500},  # Tuesday - CAP3
        {"Date": "23-Apr-26", "Volume": 2712500},  # Thursday - CAP3
        {"Date": "24-Apr-26", "Volume": 1093000},  # Friday - Germany
        # Week 17
        {"Date": "27-Apr-26", "Volume": 2712500},  # Monday - CAP3
        {"Date": "28-Apr-26", "Volume": 2712500},  # Tuesday - CAP3
        {"Date": "29-Apr-26", "Volume": 1524500},  # Wednesday - Poland
        {"Date": "30-Apr-26", "Volume": 2712500},  # Thursday - CAP3
        # ----- May 2026 -----
        # Week 17 continues
        {"Date": "01-May-26", "Volume": 1093000},  # Friday - Germany
        # Week 18
        {"Date": "04-May-26", "Volume": 2712500},  # Monday - CAP3
        {"Date": "05-May-26", "Volume": 2712500},  # Tuesday - CAP3
        {"Date": "07-May-26", "Volume": 2712500},  # Thursday - CAP3
        {"Date": "08-May-26", "Volume": 1093000},  # Friday - Germany
        # Week 19
        {"Date": "11-May-26", "Volume": 2712500},  # Monday - CAP3
        {"Date": "12-May-26", "Volume": 2712500},  # Tuesday - CAP3
        {"Date": "13-May-26", "Volume": 1524500},  # Wednesday - Poland
        {"Date": "14-May-26", "Volume": 2712500},  # Thursday - CAP3
        {"Date": "15-May-26", "Volume": 1093000},  # Friday - Germany
        # Week 20
        {"Date": "18-May-26", "Volume": 2712500},  # Monday - CAP3
        {"Date": "19-May-26", "Volume": 2712500},  # Tuesday - CAP3
        {"Date": "21-May-26", "Volume": 2712500},  # Thursday - CAP3
        {"Date": "22-May-26", "Volume": 1093000},  # Friday - Germany
        # Week 21
        {"Date": "25-May-26", "Volume": 2712500},  # Monday - CAP3
        {"Date": "26-May-26", "Volume": 2712500},  # Tuesday - CAP3
        {"Date": "27-May-26", "Volume": 1524500},  # Wednesday - Poland
        {"Date": "28-May-26", "Volume": 2712500},  # Thursday - CAP3
        {"Date": "29-May-26", "Volume": 1093000},  # Friday - Germany
        # ----- June 2026 -----
        # Week 22
        {"Date": "01-Jun-26", "Volume": 2712500},  # Monday - CAP3
        {"Date": "02-Jun-26", "Volume": 2712500},  # Tuesday - CAP3
        {"Date": "04-Jun-26", "Volume": 2712500},  # Thursday - CAP3
        {"Date": "05-Jun-26", "Volume": 1093000},  # Friday - Germany
        # Week 23
        {"Date": "08-Jun-26", "Volume": 2712500},  # Monday - CAP3
        {"Date": "09-Jun-26", "Volume": 2712500},  # Tuesday - CAP3
        {"Date": "10-Jun-26", "Volume": 1524500},  # Wednesday - Poland
        {"Date": "11-Jun-26", "Volume": 2712500},  # Thursday - CAP3
        {"Date": "12-Jun-26", "Volume": 1093000},  # Friday - Germany
        # Week 24
        {"Date": "15-Jun-26", "Volume": 2712500},  # Monday - CAP3
        {"Date": "16-Jun-26", "Volume": 2712500},  # Tuesday - CAP3
        {"Date": "18-Jun-26", "Volume": 2712500},  # Thursday - CAP3
        {"Date": "19-Jun-26", "Volume": 1093000},  # Friday - Germany
        # Week 25
        {"Date": "22-Jun-26", "Volume": 2712500},  # Monday - CAP3
        {"Date": "23-Jun-26", "Volume": 2712500},  # Tuesday - CAP3
        {"Date": "24-Jun-26", "Volume": 1524500},  # Wednesday - Poland
        {"Date": "25-Jun-26", "Volume": 2712500},  # Thursday - CAP3
        {"Date": "26-Jun-26", "Volume": 1093000},  # Friday - Germany
        # Week 26
        {"Date": "29-Jun-26", "Volume": 2712500},  # Monday - CAP3
        {"Date": "30-Jun-26", "Volume": 2712500},  # Tuesday - CAP3
        # ----- July 2026 -----
        # Week 26 continues
        {"Date": "02-Jul-26", "Volume": 2712500},  # Thursday - CAP3
        {"Date": "03-Jul-26", "Volume": 1093000},  # Friday - Germany
        # Week 27
        {"Date": "06-Jul-26", "Volume": 2712500},  # Monday - CAP3
        {"Date": "07-Jul-26", "Volume": 2712500},  # Tuesday - CAP3
        {"Date": "08-Jul-26", "Volume": 1524500},  # Wednesday - Poland
        {"Date": "09-Jul-26", "Volume": 2712500},  # Thursday - CAP3
        {"Date": "10-Jul-26", "Volume": 1093000},  # Friday - Germany
        # Week 28
        {"Date": "13-Jul-26", "Volume": 2712500},  # Monday - CAP3
        {"Date": "14-Jul-26", "Volume": 2712500},  # Tuesday - CAP3
        {"Date": "16-Jul-26", "Volume": 2712500},  # Thursday - CAP3
        {"Date": "17-Jul-26", "Volume": 1093000},  # Friday - Germany
        # Week 29
        {"Date": "20-Jul-26", "Volume": 2712500},  # Monday - CAP3
        {"Date": "21-Jul-26", "Volume": 2712500},  # Tuesday - CAP3
        {"Date": "22-Jul-26", "Volume": 1524500},  # Wednesday - Poland
        {"Date": "23-Jul-26", "Volume": 2712500},  # Thursday - CAP3
        {"Date": "24-Jul-26", "Volume": 1093000},  # Friday - Germany
        # Week 30
        {"Date": "27-Jul-26", "Volume": 2712500},  # Monday - CAP3
        {"Date": "28-Jul-26", "Volume": 2712500},  # Tuesday - CAP3
        {"Date": "30-Jul-26", "Volume": 2712500},  # Thursday - CAP3
        {"Date": "31-Jul-26", "Volume": 1093000},  # Friday - Germany
        # ----- August 2026 -----
        # Week 31
        {"Date": "03-Aug-26", "Volume": 2712500},  # Monday - CAP3
        {"Date": "04-Aug-26", "Volume": 2712500},  # Tuesday - CAP3
        {"Date": "05-Aug-26", "Volume": 1524500},  # Wednesday - Poland
        {"Date": "06-Aug-26", "Volume": 2712500},  # Thursday - CAP3
        {"Date": "07-Aug-26", "Volume": 1093000},  # Friday - Germany
        # Week 32
        {"Date": "10-Aug-26", "Volume": 2712500},  # Monday - CAP3
        {"Date": "11-Aug-26", "Volume": 2712500},  # Tuesday - CAP3
        {"Date": "13-Aug-26", "Volume": 2712500},  # Thursday - CAP3
        {"Date": "14-Aug-26", "Volume": 1093000},  # Friday - Germany
        # Week 33
        {"Date": "17-Aug-26", "Volume": 2712500},  # Monday - CAP3
        {"Date": "18-Aug-26", "Volume": 2712500},  # Tuesday - CAP3
        {
            "Date": "19-Aug-26",
            "Volume": 1525000,
        },  # Wednesday - Poland (last auction in Aug)
        {"Date": "20-Aug-26", "Volume": 2712500},  # Thursday - CAP3
        {"Date": "21-Aug-26", "Volume": 1093000},  # Friday - Germany
        # Week 34
        {"Date": "24-Aug-26", "Volume": 2712500},  # Monday - CAP3
        {"Date": "25-Aug-26", "Volume": 2712500},  # Tuesday - CAP3
        {"Date": "27-Aug-26", "Volume": 2712500},  # Thursday - CAP3
        {
            "Date": "28-Aug-26",
            "Volume": 1087500,
        },  # Friday - Germany (last auction in Aug)
        # Week 35 - Last auction in August
        {"Date": "31-Aug-26", "Volume": 2937000},  # Monday - CAP3 (last auction in Aug)
        # ----- September 2026 -----
        # Week 35 continues - Sep-Dec volumes: CAP3=3,159,000, DE=2,645,500, PL=2,848,500
        {"Date": "01-Sep-26", "Volume": 3159000},  # Tuesday - CAP3
        {"Date": "02-Sep-26", "Volume": 2848500},  # Wednesday - Poland
        {"Date": "03-Sep-26", "Volume": 3159000},  # Thursday - CAP3
        {"Date": "04-Sep-26", "Volume": 2645500},  # Friday - Germany
        # Week 36
        {"Date": "07-Sep-26", "Volume": 3159000},  # Monday - CAP3
        {"Date": "08-Sep-26", "Volume": 3159000},  # Tuesday - CAP3
        {"Date": "10-Sep-26", "Volume": 3159000},  # Thursday - CAP3
        {"Date": "11-Sep-26", "Volume": 2645500},  # Friday - Germany
        # Week 37
        {"Date": "14-Sep-26", "Volume": 3159000},  # Monday - CAP3
        {"Date": "15-Sep-26", "Volume": 3159000},  # Tuesday - CAP3
        {"Date": "16-Sep-26", "Volume": 2848500},  # Wednesday - Poland
        {"Date": "17-Sep-26", "Volume": 3159000},  # Thursday - CAP3
        {"Date": "18-Sep-26", "Volume": 2645500},  # Friday - Germany
        # Week 38
        {"Date": "21-Sep-26", "Volume": 3159000},  # Monday - CAP3
        {"Date": "22-Sep-26", "Volume": 3159000},  # Tuesday - CAP3
        {"Date": "24-Sep-26", "Volume": 3159000},  # Thursday - CAP3
        {"Date": "25-Sep-26", "Volume": 2645500},  # Friday - Germany
        # Week 39
        {"Date": "28-Sep-26", "Volume": 3159000},  # Monday - CAP3
        {"Date": "29-Sep-26", "Volume": 3159000},  # Tuesday - CAP3
        {"Date": "30-Sep-26", "Volume": 2848500},  # Wednesday - Poland
        # ----- October 2026 -----
        # Week 39 continues
        {"Date": "01-Oct-26", "Volume": 3159000},  # Thursday - CAP3
        {"Date": "02-Oct-26", "Volume": 2645500},  # Friday - Germany
        # Week 40
        {"Date": "05-Oct-26", "Volume": 3159000},  # Monday - CAP3
        {"Date": "06-Oct-26", "Volume": 3159000},  # Tuesday - CAP3
        {
            "Date": "07-Oct-26",
            "Volume": 915500,
        },  # Wednesday - Northern Ireland (special)
        {"Date": "08-Oct-26", "Volume": 3159000},  # Thursday - CAP3
        {"Date": "09-Oct-26", "Volume": 2645500},  # Friday - Germany
        # Week 41
        {"Date": "12-Oct-26", "Volume": 3159000},  # Monday - CAP3
        {"Date": "13-Oct-26", "Volume": 3159000},  # Tuesday - CAP3
        {"Date": "14-Oct-26", "Volume": 2848500},  # Wednesday - Poland
        {"Date": "15-Oct-26", "Volume": 3159000},  # Thursday - CAP3
        {"Date": "16-Oct-26", "Volume": 2645500},  # Friday - Germany
        # Week 42
        {"Date": "19-Oct-26", "Volume": 3159000},  # Monday - CAP3
        {"Date": "20-Oct-26", "Volume": 3159000},  # Tuesday - CAP3
        {"Date": "22-Oct-26", "Volume": 3159000},  # Thursday - CAP3
        {"Date": "23-Oct-26", "Volume": 2645500},  # Friday - Germany
        # Week 43
        {"Date": "26-Oct-26", "Volume": 3159000},  # Monday - CAP3
        {"Date": "27-Oct-26", "Volume": 3159000},  # Tuesday - CAP3
        {"Date": "28-Oct-26", "Volume": 2848500},  # Wednesday - Poland
        {"Date": "29-Oct-26", "Volume": 3159000},  # Thursday - CAP3
        {"Date": "30-Oct-26", "Volume": 2645500},  # Friday - Germany
        # ----- November 2026 -----
        # Week 44
        {"Date": "02-Nov-26", "Volume": 3159000},  # Monday - CAP3
        {"Date": "03-Nov-26", "Volume": 3159000},  # Tuesday - CAP3
        {"Date": "05-Nov-26", "Volume": 3159000},  # Thursday - CAP3
        {"Date": "06-Nov-26", "Volume": 2645500},  # Friday - Germany
        # Week 45
        {"Date": "09-Nov-26", "Volume": 3159000},  # Monday - CAP3
        {"Date": "10-Nov-26", "Volume": 3159000},  # Tuesday - CAP3
        {"Date": "11-Nov-26", "Volume": 2848500},  # Wednesday - Poland
        {"Date": "12-Nov-26", "Volume": 3159000},  # Thursday - CAP3
        {"Date": "13-Nov-26", "Volume": 2645500},  # Friday - Germany
        # Week 46
        {"Date": "16-Nov-26", "Volume": 3159000},  # Monday - CAP3
        {"Date": "17-Nov-26", "Volume": 3159000},  # Tuesday - CAP3
        {"Date": "19-Nov-26", "Volume": 3159000},  # Thursday - CAP3
        {"Date": "20-Nov-26", "Volume": 2645500},  # Friday - Germany
        # Week 47
        {"Date": "23-Nov-26", "Volume": 3159000},  # Monday - CAP3
        {"Date": "24-Nov-26", "Volume": 3159000},  # Tuesday - CAP3
        {"Date": "25-Nov-26", "Volume": 2848500},  # Wednesday - Poland
        {"Date": "26-Nov-26", "Volume": 3159000},  # Thursday - CAP3
        {"Date": "27-Nov-26", "Volume": 2645500},  # Friday - Germany
        # ----- December 2026 -----
        # Week 48
        {"Date": "30-Nov-26", "Volume": 3159000},  # Monday - CAP3
        {"Date": "01-Dec-26", "Volume": 3159000},  # Tuesday - CAP3
        {"Date": "03-Dec-26", "Volume": 3159000},  # Thursday - CAP3
        {"Date": "04-Dec-26", "Volume": 2645500},  # Friday - Germany
        # Week 49 - Final auctions in December
        {"Date": "07-Dec-26", "Volume": 3159000},  # Monday - CAP3
        {"Date": "08-Dec-26", "Volume": 3159000},  # Tuesday - CAP3
        {"Date": "09-Dec-26", "Volume": 2849000},  # Wednesday - Poland (final auction)
        {"Date": "10-Dec-26", "Volume": 3158000},  # Thursday - CAP3 (final auction)
        {"Date": "11-Dec-26", "Volume": 2649000},  # Friday - Germany (final auction)
    ]
)
future_calendar_eua["Date"] = pd.to_datetime(
    future_calendar_eua["Date"], format="%d-%b-%y"
)
FUTURE_CALENDAR_EUA = future_calendar_eua[
    future_calendar_eua["Date"] > pd.Timestamp.today()
]


OPTIONS = {
    "put": "cc_opt-put_ice-endex_eua-future-opt_mc_ice_d_eur/t",
    "call": "cc_opt-call_ice-endex_eua-future-opt_mc_ice_d_eur/t",
}

FUTURES = {
    "eua": "cc_mp_ice-endex_eua-future_mc_ice_d_eur/t",
    "uka": "cc_mp_ifeu_gb_uka-futures_mc_ice_d_gbp/t",
    "rggi": "cc_mp_ifus_rggi-future_mc_ice_d_usd/st",
    "wci": "cc_mp_ifus_cca-future_mc_ice_d_usd/t",
}

MARKET = {
    "ttf": "gas_mp_ice-endex_eu_ttf-future_mc_ice_d_eur/mwh",
    "coal": "coa_mp_ifeu_ara-api2-futures_mc_ice_d_usd/t",
    "germanpw_fm": "pwr_mp_eex_de_base-future_mc_eex_d_eur/mwh",
    "germanpw_fy": "pwr_mp_eex_de_base-future_yc_eex_d_eur/mwh",
    "brent": "oil_mp_ifeu_brent-crude-futures_mc_ice_d_usd/bbl",
}

SPREADS = {
    "clean_dark": "cc_spread-clean-dark-fm_a_veyt_quant-cspread_ice-eua-front-dec_medium-efficiency_d_d_eur/mwh",
    "clean_spark": "cc_spread-clean-spark-fm_a_veyt_quant-cspread_ice-eua-front-dec_medium-efficiency_d_d_eur/mwh",
    "fuel_switch": "cc_fuel-switch-price-front-month_a_veyt_quant-co2-fuel-switching-calc_d_d_eur/t",
}

AUCTIONS_EUA = {
    "DE": "cc_ar_eex_de_eua-auction_eex_w_eur/t",
    "EU": "cc_ar_eex_eu_eua-auction_eex_3w_eur/t",
    "PL": "cc_ar_eex_pl_eua-auction_eex_w2_eur/t",
}

AUCTIONS_UKA = {
    "vol": "cc_vol_ifeu_uk_uka-auction_ice_w2_w2_t",
    "ar": "cc_ar_ifeu_uk_uka-auction_ice_w2_w2_gbp/t",  # Note: tsgroup
}

AUCTIONS_RGGI = {
    "compl": "cc_no-compl-bidders_rggi-coats_rggi-auction_rggi_q",  # Note: timeseries
    "non_compl": "cc_no-non-compl-bidders_rggi-coats_rggi-auction_rggi_q",  # Note: timeseries
    "ar": "cc_ar_rggi-coats_rggi-auction_rggi_q_usd/st",  # Note: tsgroup
}

AUCTIONS_WCI = {
    "compl": "cc_pct-compl-bidders_citss_wci-curr-auction_wci_q",  # Note: timeseries
    "pcr": "cc_allowance-pcr_citss_wci-curr-auction_wci_y_usd/t",  # Note: timeseries
    "ar": "cc_ar_citss_wci-curr-auction_wci_q_usd/t",  # Note: tsgroup
}

COT = {
    "IFCI": [
        "cc_num-pos_eua-future_inv-firms-cred-inst_ice-cot_w_w_t",
        "cc_num-pers-hold-pos_eua-future_inv-firms-cred-inst_ice-cot_w_w_count",
    ],
    "IF": [
        "cc_num-pos_eua-future_inv-funds_ice-cot_w_w_t",
        "cc_num-pers-hold-pos_eua-future_inv-funds_ice-cot_w_w_count",
    ],
    "OFI": [
        "cc_num-pos_eua-future_other-inv-ints_ice-cot_w_w_t",
        "cc_num-pers-hold-pos_eua-future_other-inv-ints_ice-cot_w_w_count",
    ],
    "CU": [
        "cc_num-pos_eua-future_com-under_ice-cot_w_w_t",
        "cc_num-pers-hold-pos_eua-future_com-under_ice-cot_w_w_count",
    ],
    "OWCO": [
        "cc_num-pos_eua-future_ops-w-compl-oblig_ice-cot_w_w_t",
        "cc_num-pers-hold-pos_eua-future_ops-w-compl-oblig_ice-cot_w_w_count",
    ],
}

VOLATILITY = {
    "20d": "cc_price_a_eua-future-dec_mc_veyt_quant-future-volatility_20d-roll_d_d_eur/t",  # Note : MarketPrice
    "5d": "cc_price_a_eua-future-dec_mc_veyt_quant-future-volatility_5d-roll_d_d_eur/t",  # Note : MarketPrice
}

EUA_UKA_SPREAD = "cc_price_a_uka-eua-spread_mc_veyt_quant-future-spread_d_d_eur/t"

WEATHER = {
    "cdd": "wthr_cdd_b_eea_veyt_ecmwf_quant-weather-ec_op_ec_avg-popul_h_h12_ch",
    "hdd": "wthr_hdd_b_eea_veyt_ecmwf_quant-weather-ec_op_ec_avg-popul_h_h12_ch",
}

COLORS = [
    "#89E9CE",
    "#D3DF5E",
    "#E880D2",
    "#AEB585",
    "#E4C28C",
    "#F8F3A4",
    "#F5B19E",
    "#85ced6",
]
# COLORS = ["#b9d3de","#cfd4b9","#434d61", "#636e3d", "#E4C28C", "#F8F3A4", "#F5B19E", "#85ced6"]

# Tab colors for layout
TAB_COLOR = {"color": "white"}
ACTIVE_TAB_COLOR = {"color": "white"}

# Investment firm names for COT tab
FIRMS = [
    "Investment Firms or Credit Institutions",
    "Investment Funds",
    "Other Financial Institutions",
    "Commercial Undertakings",
    "Operators with obligations under Directive 2003/87/EC",
]

# Energy drivers labels and values
ENERGY_DRIVERS_LABEL = [
    "Brent",
    "TTF",
    "Coal",
    "German Power(front month)",
    "German Power(front year)",
    "Clean Dark",
    "Clean Spark",
    "Fuel Switch",
]

ENERGY_DRIVERS_VALUE = [
    "Brent",
    "TTF",
    "Coal",
    "GermanPower(FM)",
    "GermanPower(FY)",
    "CD",
    "CS",
    "FuelSwitch",
]

# Technical drivers labels and values
TECHNICAL_DRIVERS_LABEL = [
    "Auction-Price",
    "Auction-Ratio",
    "CoT1:Other",
    "CoT2: Operators with obligations",
    "CoT3:Investment Funds",
    "CoT4:Investment firms/credit",
    "CoT5: Commercial undertakings",
]

TECHNICAL_DRIVERS_VALUE = [
    "Auction_Price",
    "Auction_Cover_Ratio",
    "CoT_Other",
    "CoT_Obligations",
    "CoT_InvFunds",
    "CoT_InvFrims_Credit",
    "CoT_Undertakings",
]
