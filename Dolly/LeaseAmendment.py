template = "{title}\nThe parties to this {amendment} to Lease dated the {date} are {landlord} (\"Landlord\") and {tenant} (\"Tenant\").\n\
            WHEREAS, {whereas}.\n\
            The parties do {change}:\n\
            1. {sec1}\n\
            2. {sec2}\n\
            3. {sec3}\n\
            4. {sec4}\n"

variables = {
    'title': 'Wu-Hun Yang and Geoffrey Bender First Amendment to Lease Highline Professional Building',
    'amendment': 'First Amendment',
    'date': 'December 20, 2023',
    'landlord': 'Highline Professional Building Associates, Limited Partnership',
    'tenant': 'Wu-Hun Yang and Geoffrey Bender',
    'whereas': 'Landlord and Tenant have a binding Lease Agreement dated June 10, 2021, together the lease in and to the premises described in Exhibit A Legal Description of the Property more commonly known as Highline Professional Building, 16122 8th, Avenue SW, Suite D3, City of Burien, County of King, State of Washington',
    'change': 'hereby mutually agree to modify certain particulars of the Lease Agreement as follow',
    'sec1': 'Section 4 Term\nSection 4 shall be modified as follows:\n"The term of the Lease shall be extended ten (10) years commencing August 1, 2024 and ending July 31, 2034."',
    'sec2': 'Section 5 Rent.\nMonthly rent for the extended term of the lease shall be as follows:\n\
        Months Total  Monthly Rent\n\
        August 1, 2024 - July 31, 2025  $ 2600 per month\n\
        August 1, 2025 - July 31, 2026  $ 2800 per month\n\
        August 1, 2026 - July 31, 2027  $ 3000 per month\n\
        August 1, 2027 - July 31, 2028  $ 3090 per month\n\
        August 1, 2028 - July 31, 2029  $ 3183 per month\n\
        August 1, 2029 - July 31, 2030  $ 3278 per month\n\
        August 1, 2030 - July 31, 2031  $ 3377 per month\n\
        August 1, 2031 - July 31, 2032  $ 3478 per month\n\
        August 1, 2032 - July 31, 2033  $ 3582 per month\n\
        August 1, 2033 - July 31, 2034  $ 3690 per month',
    'sec3': 'Tenant Improvements. The following Tenant Improvements will be handled by Landlord at Landlord\'s cost and expense.\n\
        - Replace the sinks in the bathrooms and update the patient bathroom\n\
        - Replace the cabinets in our treatment rooms with new and replace the rubber base as it isn\'t flush\n\
        - Touch up paint as needed\n\
        - Add sinks to the two rooms that do not currently have sinks, replace carpet in one of them with a liquid resistant floor\n\
        - doors/trims of the treatment rooms and the older wood on patient facing common area surfaces be sanded/painted',
    'sec4': 'Binding Effect.\The Lease, as hereby amended, shall continue in full force and effect subject to the terms and provisions thereof and hereof.  This First Amendment to Lease shall be binding upon and inure to the benefit of the Lessor and Lessee, and their respective successors and permitted assigns.  In the event of any conflict between the terms, covenants, and conditions of the First Amendment to Lease, the terms, covenants, and conditions of this First Amendment to Lease shall continue.\n\
        REPRESENTATION: Azose Commercial Properties (“Broker”) represent the ownership in this transaction.  Tenant confirms receipt of “the Law of Real Estate Agency” as required under RCW 18.86.030.\n\
        IN WITNESS WHEREOF, Landlord and Tenant have caused this First Amendment to Lease to be executed by their duly authorized offices as of the day and year first written above.\n\
        Dated: December 20, 2023\n\
        TENANT:     LANDLORD\n\
        Wu-Hsun Yang and Geoffrey Bender     Highline Professional Building Association Limited Partnership, a Washington limited partnership'
}

lease = str.format(template, **variables)
print(lease)