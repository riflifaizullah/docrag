"""English content for the sample PDFs. Same structure and the same facts as the Indonesian set, in dollars."""

COMPANY = "Example Corp"

NOTICE = (
    "This document is a fictional sample for demonstrating DocRAG. The company, "
    "policies and figures in it are not real."
)

FOOTER = "Page {page} of {total}"

DOCS = [
    (
        "Employee_Handbook_Sample.pdf",
        "Employee Handbook",
        "No. EH-001/2026",
        [
            ("CHAPTER I GENERAL PROVISIONS", [
                ("Article 1 Purpose", ["This handbook sets out the rights and duties of employees of Example Corp so that the working relationship stays orderly, fair and transparent."]),
                ("Article 2 Scope", ["This handbook applies to all permanent and contract employees in every business unit. Special terms for contract employees are set out in their individual employment agreements."]),
            ]),
            ("CHAPTER II WORKING HOURS", [
                ("Article 3 Working Days and Hours", ["Working days are Monday to Friday. Working hours start at 08:00 and end at 17:00, with a one hour break at 12:00.", "Employees may work from home at most two days a week with the approval of their direct supervisor."]),
                ("Article 4 Overtime", ["Overtime must be approved in writing by the supervisor before it is worked. Overtime is compensated with overtime pay or time off in lieu, as agreed."]),
            ]),
            ("CHAPTER III LEAVE", [
                ("Article 5 Annual Leave", ["Permanent employees are entitled to 12 days of annual leave after completing one year of service. Details and the request procedure are set out in the Leave Policy."]),
                ("Article 6 Other Leave", ["Employees are entitled to sick leave, maternity leave and leave for important personal reasons in line with company rules."]),
            ]),
            ("CHAPTER IV DISCIPLINE", [
                ("Article 7 Employee Duties", ["Employees must protect company data, use office facilities reasonably and report conflicts of interest to their supervisor."]),
                ("Article 8 Sanctions", ["Breaches of this handbook may lead to a verbal warning, a written warning or other action in proportion to the breach, after the employee has had a chance to explain."]),
            ]),
            (None, [
                ("Appendix 1 Unit Contact List", ["Questions about this handbook can be sent to the Human Resources team by email at hr@example.example or on extension 101 on working days."]),
            ]),
        ],
    ),
    (
        "Leave_Policy_Sample.pdf",
        "Leave Policy",
        "No. LP-002/2026",
        [
            ("CHAPTER I GENERAL PROVISIONS", [
                ("Article 1 Definitions", ["Leave is an absence from work that the company has approved. A working day is Monday to Friday, excluding public holidays."]),
            ]),
            ("CHAPTER II TYPES OF LEAVE", [
                ("Article 2 Annual Leave", ["Permanent employees are entitled to 12 days of annual leave after completing one year of service.", "Unused annual leave can be carried over to the next year up to 6 days. Leave above that limit lapses at the end of March."]),
                ("Article 3 Requesting Leave", ["Annual leave must be requested at least 3 working days before the leave date through the direct supervisor, using the form in the appendix to this policy.", "The supervisor decides within 2 working days of receiving the request."]),
                ("Article 4 Sick Leave", ["An employee who is sick for one or two days only needs to tell the supervisor. Sickness of more than two consecutive days must be supported by a doctor's note."]),
                ("Article 5 Maternity Leave", ["A mother is entitled to 3 months of maternity leave on full pay. A father is entitled to 5 working days of paternity leave."]),
                ("Article 6 Leave for Important Reasons", ["Leave for important reasons is granted for at most 3 working days for the employee's own wedding, 2 working days for a child's wedding and 3 working days for the death of an immediate family member."]),
            ]),
            (None, [
                ("Appendix 1 Leave Request Form", ["The form contains the employee's name, business unit, type of leave, start and end dates, the name of the colleague covering the duties, and the approval of the supervisor and the Human Resources team."]),
                ("Appendix 2 Summary of Leave Entitlements", ["Annual leave 12 days, sick leave as per the doctor's note, maternity leave 3 months, paternity leave 5 working days and wedding leave 3 working days."]),
            ]),
        ],
    ),
    (
        "Business_Travel_Policy_Sample.pdf",
        "Business Travel Policy",
        "No. BT-003/2026",
        [
            ("CHAPTER I GENERAL PROVISIONS", [
                ("Article 1 Purpose", ["This policy sets out the approval, costs and accountability for business travel by employees of Example Corp."]),
                ("Article 2 Travel Approval", ["Business travel must be approved by the supervisor and the Finance team at least 5 working days before departure."]),
            ]),
            ("CHAPTER II TRAVEL COSTS", [
                ("Article 3 Modes of Transport", ["Air travel is economy class. Ground travel uses public transport or a company vehicle.", "Taxis are limited to trips to and from the airport or station."]),
                ("Article 4 Daily Allowance", ["The daily allowance for domestic business travel is $30 per day and covers meals and other small costs.", "Hotel accommodation is reimbursed against receipts up to $60 per night."]),
                ("Article 5 Expense Claims", ["Travel expense claims must be submitted within 7 working days of the employee's return, with the original receipts attached. Late claims are processed in the next payment period."]),
            ]),
            (None, [
                ("Appendix 1 Travel Expense Claim Form", ["The form contains the purpose of the trip, departure and return dates, a breakdown of costs by category, and the supervisor's signature."]),
            ]),
        ],
    ),
    (
        "Information_Security_Policy_Sample.pdf",
        "Information Security Policy",
        "No. IS-004/2026",
        [
            ("CHAPTER I GENERAL PROVISIONS", [
                ("Article 1 Purpose", ["This policy protects the data and information systems of Example Corp from unauthorized access, loss and misuse."]),
            ]),
            ("CHAPTER II PROTECTING ACCOUNTS AND DEVICES", [
                ("Article 2 Passwords", ["Passwords must be at least 12 characters long, include upper case letters, lower case letters, numbers and symbols, and be changed at least every 90 days.", "Passwords must not be shared with anyone or written down in plain sight."]),
                ("Article 3 Devices", ["Company laptops must use disk encryption and lock the screen automatically after 5 minutes of inactivity. Using a personal device to access company data requires the approval of the information security team."]),
            ]),
            ("CHAPTER III DATA AND INCIDENTS", [
                ("Article 4 Data Classification", ["Data is classified as public, internal or confidential. Confidential data may only be sent through approved encrypted channels."]),
                ("Article 5 Incident Reporting", ["Any suspected information security incident must be reported to the information security team within 24 hours of being noticed."]),
            ]),
            (None, [
                ("Appendix 1 Incident Response Contacts", ["The information security team can be reached by email at security@example.example or on extension 199 every working day from 08:00 to 17:00."]),
            ]),
        ],
    ),
    (
        "Procurement_Procedure_Sample.pdf",
        "Procurement Procedure for Goods and Services",
        "No. PR-005/2026",
        [
            ("CHAPTER I GENERAL PROVISIONS", [
                ("Article 1 Purpose", ["This procedure makes sure that Example Corp buys goods and services fairly, with proper records and free from conflicts of interest."]),
            ]),
            ("CHAPTER II METHODS AND APPROVAL", [
                ("Article 2 Value Limits and Methods", ["Purchases of up to $1,000 can be made by direct purchase.", "Purchases above $1,000 and up to $10,000 must compare at least 3 quotations.", "Purchases above $10,000 are made through a limited tender or an open tender."]),
                ("Article 3 Approval", ["A purchase request must be approved by the head of the unit and the Finance team. Purchases above $10,000 also need the approval of a director."]),
            ]),
            ("CHAPTER III SUPPLIERS", [
                ("Article 4 Supplier Evaluation", ["Suppliers are evaluated on price, quality, punctuality and administrative compliance. A supplier with a record of repeated delays can be removed from the supplier list."]),
                ("Article 5 Conflicts of Interest", ["An employee who has a family relationship or financial interest with a supplier must declare it and must not take part in the evaluation."]),
            ]),
            (None, [
                ("Appendix 1 Procurement Document Checklist", ["The documents that must be complete: the purchase request, the quotation comparison where required, approval according to the value limit, the contract or purchase order, and the handover record."]),
                ("Appendix 2 Handover Record Format", ["The handover record contains the handover date, details of the goods or services, their condition when received, and the signatures of the recipient and the supplier."]),
            ]),
        ],
    ),
]
