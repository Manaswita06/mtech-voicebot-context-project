# ============================================================
# lexicon.py
#
# Scenario-specific lexicon for synthetic enterprise
# customer-support conversations.
#
# Each scenario contains:
# - scenario family
# - valid entities
# - logical dialogue flow
# - customer templates
# - agent templates
#
# IMPORTANT:
# Templates are intentionally scoped to each scenario.
# ============================================================

SCENARIOS = {

    # ========================================================
    # CARD MANAGEMENT
    # ========================================================

    "CARD_REPLACEMENT": {
        "family": "Card Management",

        "entities": {
            "card_issue": [
                "damaged magnetic strip",
                "damaged chip",
                "physically damaged card",
                "worn-out card",
                "card that is no longer working"
            ],
            "verification": [
                "the last four digits of my card are 4821",
                "the last four digits are 7316",
                "the card ends in 9024"
            ],
            "delivery_address": [
                "my registered address",
                "the address currently associated with my account"
            ],
            "delivery_timeline": [
                "three to five working days",
                "five to seven working days",
                "up to seven business days"
            ]
        },

        "flow": [
            "problem_statement",
            "acknowledge",
            "verification_request",
            "verification_response",
            "address_confirmation",
            "address_response",
            "resolve",
            "inform",
            "confirm",
            "closing"
        ],

        "templates": {
            "problem_statement": {
                "customer": [
                    "Hi, my card has a {card_issue}. Can you send me a replacement?",
                    "My card is not working because of a {card_issue}. I would like a replacement.",
                    "I need to replace my card because it has a {card_issue}."
                ]
            },
            "acknowledge": {
                "agent": [
                    "I understand. I can help you arrange a replacement card.",
                    "I am sorry you are experiencing this issue. Let us get a replacement arranged."
                ]
            },
            "verification_request": {
                "agent": [
                    "For verification, could you please confirm the last four digits of the card?",
                    "Before I process the replacement, please confirm the last four digits of your card."
                ]
            },
            "verification_response": {
                "customer": [
                    "{verification}.",
                    "Sure, {verification}."
                ]
            },
            "address_confirmation": {
                "agent": [
                    "Would you like the replacement sent to {delivery_address}?",
                    "Can I send the replacement card to {delivery_address}?"
                ]
            },
            "address_response": {
                "customer": [
                    "Yes, please send it to {delivery_address}.",
                    "Yes, that address is correct."
                ]
            },
            "resolve": {
                "agent": [
                    "Your replacement card request has been submitted successfully.",
                    "I have successfully initiated the replacement request."
                ]
            },
            "inform": {
                "agent": [
                    "Your replacement card should arrive within {delivery_timeline}.",
                    "The expected delivery time is {delivery_timeline}."
                ]
            },
            "confirm": {
                "customer": [
                    "That is fine. Thank you for your help.",
                    "Great, thank you for arranging the replacement."
                ]
            },
            "closing": {
                "agent": [
                    "You are welcome. Please let us know if you need further assistance.",
                    "Happy to help. Have a good day."
                ]
            }
        }
    },

    "CARD_ACTIVATION": {
        "family": "Card Management",

        "entities": {
            "verification": [
                "the last four digits are 4821",
                "my card ends in 7316",
                "the card ending is 9024"
            ]
        },

        "flow": [
            "problem_statement",
            "acknowledge",
            "verification_request",
            "verification_response",
            "resolve",
            "inform",
            "confirm",
            "closing"
        ],

        "templates": {
            "problem_statement": {
                "customer": [
                    "I received my new card but I need help activating it.",
                    "My new card has arrived and I am unable to activate it.",
                    "Can you help me activate my replacement card?"
                ]
            },
            "acknowledge": {
                "agent": [
                    "Certainly. I can help you activate your card.",
                    "I understand. Let us complete the card activation process."
                ]
            },
            "verification_request": {
                "agent": [
                    "For security, please confirm the last four digits of your card.",
                    "Could you verify the last four digits of the card you want to activate?"
                ]
            },
            "verification_response": {
                "customer": [
                    "{verification}.",
                    "Yes, {verification}."
                ]
            },
            "resolve": {
                "agent": [
                    "Your card has been activated successfully.",
                    "I have completed the activation process for your card."
                ]
            },
            "inform": {
                "agent": [
                    "You can now use your card for eligible transactions.",
                    "The card is active and ready to use."
                ]
            },
            "confirm": {
                "customer": [
                    "Great, thank you.",
                    "That worked. Thank you for your help."
                ]
            },
            "closing": {
                "agent": [
                    "You are welcome. Is there anything else I can help you with today?",
                    "Happy to help. Have a good day."
                ]
            }
        }
    },

    "CARD_BLOCK": {
        "family": "Card Security",

        "entities": {
            "reason": [
                "I cannot find my card",
                "I believe my card may have been stolen",
                "I noticed suspicious activity"
            ]
        },

        "flow": [
            "problem_statement",
            "acknowledge",
            "verification_request",
            "verification_response",
            "resolve",
            "inform",
            "confirm",
            "closing"
        ],

        "templates": {
            "problem_statement": {
                "customer": [
                    "I need to block my card because {reason}.",
                    "{reason}. Please block my card immediately.",
                    "Can you help me block my card? {reason}."
                ]
            },
            "acknowledge": {
                "agent": [
                    "I understand. I will help you secure the card.",
                    "I am sorry to hear that. Let us block the card immediately."
                ]
            },
            "verification_request": {
                "agent": [
                    "For security, please complete the required verification.",
                    "Before I block the card, I need to verify your identity."
                ]
            },
            "verification_response": {
                "customer": [
                    "Sure, I can complete the verification.",
                    "Yes, please proceed with verification."
                ]
            },
            "resolve": {
                "agent": [
                    "Your card has now been blocked successfully.",
                    "I have secured and blocked your card."
                ]
            },
            "inform": {
                "agent": [
                    "The card can no longer be used for new transactions.",
                    "Your account is now protected from further use of that card."
                ]
            },
            "confirm": {
                "customer": [
                    "Thank you for doing that quickly.",
                    "Okay, thank you for your help."
                ]
            },
            "closing": {
                "agent": [
                    "You are welcome. Please contact us if you need any further assistance.",
                    "Happy to help. Have a good day."
                ]
            }
        }
    },

    "CARD_UNBLOCK": {
        "family": "Card Security",

        "entities": {
            "reason": [
                "I found my card",
                "the security concern has been resolved",
                "I accidentally blocked my card"
            ]
        },

        "flow": [
            "problem_statement",
            "acknowledge",
            "verification_request",
            "verification_response",
            "status_check",
            "resolve",
            "confirm",
            "closing"
        ],

        "templates": {
            "problem_statement": {
                "customer": [
                    "My card is blocked and {reason}. Can it be unblocked?",
                    "{reason}. I would like to unblock my card.",
                    "Can you help me unblock my card?"
                ]
            },
            "acknowledge": {
                "agent": [
                    "Certainly. I can check whether the card is eligible to be unblocked.",
                    "I understand. Let me review the card status."
                ]
            },
            "verification_request": {
                "agent": [
                    "For security, please complete account verification.",
                    "Before making changes, I need to verify your identity."
                ]
            },
            "verification_response": {
                "customer": [
                    "Sure, I can verify my details.",
                    "Yes, please proceed."
                ]
            },
            "status_check": {
                "agent": [
                    "I have reviewed the current card status.",
                    "Let me check whether there are any restrictions on the card."
                ]
            },
            "resolve": {
                "agent": [
                    "The card has been unblocked successfully.",
                    "I have removed the block from your card."
                ]
            },
            "confirm": {
                "customer": [
                    "Great, thank you.",
                    "Thank you for your help."
                ]
            },
            "closing": {
                "agent": [
                    "You are welcome. Please contact us again if you experience any issues.",
                    "Happy to help. Have a good day."
                ]
            }
        }
    },

    "CARD_PIN_RESET": {
        "family": "Card Management",

        "entities": {
            "channel": [
                "mobile banking app",
                "online banking website",
                "ATM PIN management service"
            ]
        },

        "flow": [
            "problem_statement",
            "acknowledge",
            "verification_request",
            "verification_response",
            "resolve",
            "inform",
            "confirm",
            "closing"
        ],

        "templates": {
            "problem_statement": {
                "customer": [
                    "I forgot my card PIN and need to reset it.",
                    "Can you help me reset my card PIN?",
                    "I need assistance changing my card PIN."
                ]
            },
            "acknowledge": {
                "agent": [
                    "Certainly. I can help you with the PIN reset process.",
                    "I understand. Let us help you reset your PIN."
                ]
            },
            "verification_request": {
                "agent": [
                    "For security, please complete identity verification.",
                    "Before proceeding, I need to verify your account."
                ]
            },
            "verification_response": {
                "customer": [
                    "Sure, I can complete the verification.",
                    "Yes, please proceed."
                ]
            },
            "resolve": {
                "agent": [
                    "Your PIN reset request has been processed.",
                    "I have successfully initiated the PIN reset."
                ]
            },
            "inform": {
                "agent": [
                    "You can complete the new PIN setup through the {channel}.",
                    "Please use the {channel} to create your new PIN."
                ]
            },
            "confirm": {
                "customer": [
                    "Okay, thank you.",
                    "Great, I understand."
                ]
            },
            "closing": {
                "agent": [
                    "You are welcome. Please contact us if you need further assistance.",
                    "Happy to help. Have a good day."
                ]
            }
        }
    },

    "CARD_DELIVERY_STATUS": {
        "family": "Card Delivery",

        "entities": {
            "delivery_status": [
                "currently being processed",
                "out for delivery",
                "scheduled for dispatch"
            ],
            "delivery_timeline": [
                "one to two working days",
                "three to five working days",
                "up to seven working days"
            ]
        },

        "flow": [
            "problem_statement",
            "acknowledge",
            "verification_request",
            "verification_response",
            "status_check",
            "inform",
            "confirm",
            "closing"
        ],

        "templates": {
            "problem_statement": {
                "customer": [
                    "Can you check the delivery status of my card?",
                    "I would like to know where my new card is.",
                    "Can you tell me when my card will arrive?"
                ]
            },
            "acknowledge": {
                "agent": [
                    "Certainly. I can check the card delivery status for you.",
                    "I understand. Let me look up the delivery information."
                ]
            },
            "verification_request": {
                "agent": [
                    "Please complete verification before I access the delivery details.",
                    "For security, I need to verify your account."
                ]
            },
            "verification_response": {
                "customer": [
                    "Sure, please proceed.",
                    "Yes, I can verify my details."
                ]
            },
            "status_check": {
                "agent": [
                    "I have checked the delivery tracking information.",
                    "Let me review the current delivery status."
                ]
            },
            "inform": {
                "agent": [
                    "Your card is {delivery_status} and should arrive within {delivery_timeline}.",
                    "The current status is {delivery_status}. Expected delivery is {delivery_timeline}."
                ]
            },
            "confirm": {
                "customer": [
                    "Thank you for checking.",
                    "Okay, that is helpful."
                ]
            },
            "closing": {
                "agent": [
                    "You are welcome. Please contact us if you need further assistance.",
                    "Happy to help. Have a good day."
                ]
            }
        }
    },

    "CARD_NOT_RECEIVED": {
        "family": "Card Delivery",

        "entities": {
            "expected_period": [
                "last week",
                "a few days ago",
                "earlier this month"
            ]
        },

        "flow": [
            "problem_statement",
            "acknowledge",
            "verification_request",
            "verification_response",
            "status_check",
            "resolve",
            "inform",
            "closing"
        ],

        "templates": {
            "problem_statement": {
                "customer": [
                    "My card was expected {expected_period}, but I have not received it.",
                    "I still have not received my card.",
                    "Can you help me because my replacement card has not arrived?"
                ]
            },
            "acknowledge": {
                "agent": [
                    "I am sorry to hear that. Let me check the delivery status.",
                    "I understand your concern. I will review the card delivery details."
                ]
            },
            "verification_request": {
                "agent": [
                    "Please complete account verification before I access the delivery information.",
                    "For security, I need to verify your account."
                ]
            },
            "verification_response": {
                "customer": [
                    "Sure, I can verify my details.",
                    "Yes, please proceed."
                ]
            },
            "status_check": {
                "agent": [
                    "I have checked the delivery record.",
                    "Let me review the current shipment status."
                ]
            },
            "resolve": {
                "agent": [
                    "I have raised a follow-up request regarding the missing card.",
                    "A delivery investigation has been initiated."
                ]
            },
            "inform": {
                "agent": [
                    "You will receive an update once the delivery team completes the review.",
                    "We will notify you about the next steps."
                ]
            },
            "closing": {
                "agent": [
                    "Thank you for your patience. Please contact us if you need further assistance.",
                    "We will keep you updated regarding the card delivery."
                ]
            }
        }
    },

    "CARD_LIMIT_INCREASE": {
        "family": "Card Management",

        "entities": {
            "requested_limit": [
                "a higher spending limit",
                "an increased credit limit",
                "a temporary limit increase"
            ]
        },

        "flow": [
            "problem_statement",
            "acknowledge",
            "verification_request",
            "verification_response",
            "eligibility_check",
            "inform",
            "confirm",
            "closing"
        ],

        "templates": {
            "problem_statement": {
                "customer": [
                    "I would like to request {requested_limit}.",
                    "Can you help me apply for {requested_limit}?",
                    "I need {requested_limit} on my card."
                ]
            },
            "acknowledge": {
                "agent": [
                    "Certainly. I can help you check your eligibility.",
                    "I understand. Let us review the available options."
                ]
            },
            "verification_request": {
                "agent": [
                    "Please complete account verification before I review your request.",
                    "For security, I need to verify your identity."
                ]
            },
            "verification_response": {
                "customer": [
                    "Sure, please proceed.",
                    "Yes, I can complete the verification."
                ]
            },
            "eligibility_check": {
                "agent": [
                    "I have reviewed your account for limit increase eligibility.",
                    "Let me check the account eligibility criteria."
                ]
            },
            "inform": {
                "agent": [
                    "Your request for {requested_limit} has been submitted for review.",
                    "The request will be evaluated according to account eligibility."
                ]
            },
            "confirm": {
                "customer": [
                    "Okay, thank you for checking.",
                    "Thank you. I understand."
                ]
            },
            "closing": {
                "agent": [
                    "You are welcome. We will notify you once the review is complete.",
                    "Happy to help. Have a good day."
                ]
            }
        }
    },

    "CARD_LIMIT_DECREASE": {
        "family": "Card Management",

        "entities": {
            "requested_limit": [
                "a lower spending limit",
                "a reduced card limit",
                "a lower credit limit"
            ]
        },

        "flow": [
            "problem_statement",
            "acknowledge",
            "verification_request",
            "verification_response",
            "resolve",
            "confirm",
            "closing"
        ],

        "templates": {
            "problem_statement": {
                "customer": [
                    "I would like to set {requested_limit}.",
                    "Can you help me request {requested_limit}?",
                    "I want to reduce my current card limit."
                ]
            },
            "acknowledge": {
                "agent": [
                    "Certainly. I can help you with that request.",
                    "I understand. Let us review the limit change."
                ]
            },
            "verification_request": {
                "agent": [
                    "For security, please complete account verification.",
                    "Before changing the limit, I need to verify your identity."
                ]
            },
            "verification_response": {
                "customer": [
                    "Sure, please proceed.",
                    "Yes, I can verify my details."
                ]
            },
            "resolve": {
                "agent": [
                    "Your request for {requested_limit} has been processed.",
                    "I have successfully submitted the limit reduction request."
                ]
            },
            "confirm": {
                "customer": [
                    "Thank you for your help.",
                    "Great, thank you."
                ]
            },
            "closing": {
                "agent": [
                    "You are welcome. Is there anything else I can help you with?",
                    "Happy to help. Have a good day."
                ]
            }
        }
    },

    "CASH_WITHDRAWAL_ISSUE": {
        "family": "Cash Withdrawal",

        "entities": {
            "issue": [
                "the ATM did not dispense cash",
                "the ATM dispensed the wrong amount",
                "the transaction failed but my account was debited"
            ],
            "location": [
                "a nearby ATM",
                "an ATM outside my bank branch",
                "a local cash machine"
            ]
        },

        "flow": [
            "problem_statement",
            "acknowledge",
            "probing_question",
            "customer_response",
            "verification_request",
            "verification_response",
            "resolve",
            "inform",
            "confirm",
            "closing"
        ],

        "templates": {
            "problem_statement": {
                "customer": [
                    "I had a problem because {issue}.",
                    "There was an ATM issue. {issue}.",
                    "I need help because {issue}."
                ]
            },
            "acknowledge": {
                "agent": [
                    "I understand. I will help you review the cash withdrawal issue.",
                    "I am sorry you experienced that problem. Let us check it."
                ]
            },
            "probing_question": {
                "agent": [
                    "Did this happen at {location}?",
                    "Could you tell me where the ATM transaction occurred?"
                ]
            },
            "customer_response": {
                "customer": [
                    "Yes, it happened at {location}.",
                    "I was using {location}."
                ]
            },
            "verification_request": {
                "agent": [
                    "Please complete verification before I review the transaction.",
                    "For security, I need to verify your account."
                ]
            },
            "verification_response": {
                "customer": [
                    "Sure, please proceed.",
                    "Yes, I can verify my details."
                ]
            },
            "resolve": {
                "agent": [
                    "I have registered the cash withdrawal issue for review.",
                    "The transaction has been flagged for investigation."
                ]
            },
            "inform": {
                "agent": [
                    "You will receive an update once the review is completed.",
                    "The relevant team will process the transaction review."
                ]
            },
            "confirm": {
                "customer": [
                    "Okay, thank you.",
                    "Thank you for your help."
                ]
            },
            "closing": {
                "agent": [
                    "You are welcome. Please contact us if you need further assistance.",
                    "Happy to help. Have a good day."
                ]
            }
        }
    },

    # ========================================================
    # PAYMENTS
    # ========================================================

    "PAYMENT_FAILED": {
        "family": "Payments",

        "entities": {
            "payment_channel": [
                "an online store",
                "a local merchant",
                "the mobile app"
            ]
        },

        "flow": [
            "problem_statement",
            "acknowledge",
            "probing_question",
            "customer_response",
            "status_check",
            "resolve",
            "confirm",
            "closing"
        ],

        "templates": {
            "problem_statement": {
                "customer": [
                    "My payment failed and did not go through.",
                    "I tried to make a payment but it was declined.",
                    "I am unable to complete a payment."
                ]
            },
            "acknowledge": {
                "agent": [
                    "I understand. Let me help you check the payment issue.",
                    "I am sorry you are experiencing this problem."
                ]
            },
            "probing_question": {
                "agent": [
                    "Were you making the payment through {payment_channel}?",
                    "Could you tell me where you attempted the payment?"
                ]
            },
            "customer_response": {
                "customer": [
                    "Yes, I was using {payment_channel}.",
                    "The payment attempt was through {payment_channel}."
                ]
            },
            "status_check": {
                "agent": [
                    "I have reviewed the current payment status.",
                    "Let me check whether there is any restriction affecting the transaction."
                ]
            },
            "resolve": {
                "agent": [
                    "The necessary account checks have been completed. Please try the payment again.",
                    "The payment issue has been reviewed and you can retry the transaction."
                ]
            },
            "confirm": {
                "customer": [
                    "Okay, I will try again.",
                    "Thank you. I will retry the payment."
                ]
            },
            "closing": {
                "agent": [
                    "You are welcome. Please contact us again if the issue continues.",
                    "Happy to help. Have a good day."
                ]
            }
        }
    },

    "PAYMENT_PENDING": {
        "family": "Payments",

        "entities": {
            "payment_time": [
                "a few minutes ago",
                "earlier today",
                "yesterday"
            ]
        },

        "flow": [
            "problem_statement",
            "acknowledge",
            "verification_request",
            "verification_response",
            "status_check",
            "inform",
            "confirm",
            "closing"
        ],

        "templates": {
            "problem_statement": {
                "customer": [
                    "My payment is still pending.",
                    "I made a payment but the status has not changed from pending.",
                    "Can you check why my transaction is pending?"
                ]
            },
            "acknowledge": {
                "agent": [
                    "Certainly. I can check the payment status.",
                    "I understand your concern. Let me review the transaction."
                ]
            },
            "verification_request": {
                "agent": [
                    "Could you confirm when you made the payment?",
                    "Please tell me approximately when the transaction was attempted."
                ]
            },
            "verification_response": {
                "customer": [
                    "I made the payment {payment_time}.",
                    "The transaction was attempted {payment_time}."
                ]
            },
            "status_check": {
                "agent": [
                    "I have checked the transaction and it is still being processed.",
                    "The payment is currently pending in the processing system."
                ]
            },
            "inform": {
                "agent": [
                    "Please allow some additional time for the transaction to complete.",
                    "The status should update automatically once processing is completed."
                ]
            },
            "confirm": {
                "customer": [
                    "Okay, I understand.",
                    "Thank you for checking."
                ]
            },
            "closing": {
                "agent": [
                    "You are welcome. Please contact us again if the status does not update.",
                    "Happy to help. Have a good day."
                ]
            }
        }
    },

    "PAYMENT_REVERSED": {
        "family": "Payments",

        "entities": {
            "transaction_type": [
                "a card payment",
                "an online purchase",
                "a merchant transaction"
            ]
        },

        "flow": [
            "problem_statement",
            "acknowledge",
            "verification_request",
            "verification_response",
            "status_check",
            "inform",
            "confirm",
            "closing"
        ],

        "templates": {
            "problem_statement": {
                "customer": [
                    "My {transaction_type} was reversed. Can you explain why?",
                    "I noticed that {transaction_type} was reversed.",
                    "Why was my payment reversed?"
                ]
            },
            "acknowledge": {
                "agent": [
                    "I understand. Let me review the transaction status.",
                    "Certainly. I can help explain the reversal."
                ]
            },
            "verification_request": {
                "agent": [
                    "Please provide the approximate transaction details.",
                    "Could you confirm the date or approximate time of the transaction?"
                ]
            },
            "verification_response": {
                "customer": [
                    "It happened earlier today.",
                    "The transaction was made recently."
                ]
            },
            "status_check": {
                "agent": [
                    "I have reviewed the transaction and confirmed the reversal status.",
                    "The transaction record shows that the payment was reversed."
                ]
            },
            "inform": {
                "agent": [
                    "The funds should return to your available balance according to the normal processing timeline.",
                    "The reversed amount will be reflected once the transaction update is completed."
                ]
            },
            "confirm": {
                "customer": [
                    "Okay, thank you for explaining.",
                    "That answers my question. Thank you."
                ]
            },
            "closing": {
                "agent": [
                    "You are welcome. Please contact us if you have further questions.",
                    "Happy to help. Have a good day."
                ]
            }
        }
    },

    "DUPLICATE_CHARGE": {
        "family": "Payments",

        "entities": {
            "merchant": [
                "an online retailer",
                "a grocery store",
                "a subscription service"
            ]
        },

        "flow": [
            "problem_statement",
            "acknowledge",
            "verification_request",
            "verification_response",
            "status_check",
            "resolve",
            "inform",
            "closing"
        ],

        "templates": {
            "problem_statement": {
                "customer": [
                    "I think I was charged twice by {merchant}.",
                    "There appears to be a duplicate charge from {merchant}.",
                    "I see two charges for the same purchase."
                ]
            },
            "acknowledge": {
                "agent": [
                    "I understand your concern. Let me review the transactions.",
                    "I can help you investigate the duplicate charge."
                ]
            },
            "verification_request": {
                "agent": [
                    "Could you confirm the approximate transaction date?",
                    "Please provide the transaction details you are referring to."
                ]
            },
            "verification_response": {
                "customer": [
                    "The charges appeared today.",
                    "The duplicate transactions were made recently."
                ]
            },
            "status_check": {
                "agent": [
                    "I have reviewed the transaction records.",
                    "Let me check whether both charges have been fully processed."
                ]
            },
            "resolve": {
                "agent": [
                    "The duplicate charge issue has been registered for review.",
                    "I have initiated the appropriate transaction review."
                ]
            },
            "inform": {
                "agent": [
                    "You will receive an update once the review is completed.",
                    "We will notify you about the outcome of the transaction review."
                ]
            },
            "closing": {
                "agent": [
                    "Thank you for reporting the issue.",
                    "Please contact us again if you need further assistance."
                ]
            }
        }
    },

    "UNRECOGNIZED_TRANSACTION": {
        "family": "Fraud",

        "entities": {
            "transaction": [
                "a charge I do not recognize",
                "an unfamiliar payment",
                "a suspicious transaction"
            ]
        },

        "flow": [
            "problem_statement",
            "acknowledge",
            "transaction_verification",
            "customer_response",
            "security_action",
            "resolve",
            "inform",
            "closing"
        ],

        "templates": {
            "problem_statement": {
                "customer": [
                    "I noticed {transaction} on my account.",
                    "There is {transaction} and I do not think I made it.",
                    "Can you help me with {transaction}?"
                ]
            },
            "acknowledge": {
                "agent": [
                    "I understand your concern. I will help you review the transaction.",
                    "I am sorry to hear that. Let us investigate it."
                ]
            },
            "transaction_verification": {
                "agent": [
                    "Can you confirm that you did not make or authorize this transaction?",
                    "Please confirm that you do not recognize this payment."
                ]
            },
            "customer_response": {
                "customer": [
                    "Yes, I did not make this transaction.",
                    "That is correct. I do not recognize the charge."
                ]
            },
            "security_action": {
                "agent": [
                    "I have taken the necessary security steps to protect the account.",
                    "The account has been secured while the transaction is reviewed."
                ]
            },
            "resolve": {
                "agent": [
                    "I have registered the transaction for investigation.",
                    "The unauthorized transaction review has been initiated."
                ]
            },
            "inform": {
                "agent": [
                    "You will receive updates as the investigation progresses.",
                    "We will notify you once there is an update."
                ]
            },
            "closing": {
                "agent": [
                    "Please continue monitoring your account for unfamiliar activity.",
                    "Thank you for reporting this issue promptly."
                ]
            }
        }
    },

    "PAYMENT_CONFIRMATION": {
        "family": "Payments",

        "entities": {
            "transaction": [
                "a recent payment",
                "an online purchase",
                "a card transaction"
            ]
        },

        "flow": [
            "problem_statement",
            "acknowledge",
            "verification_request",
            "verification_response",
            "inform",
            "confirm",
            "closing"
        ],

        "templates": {
            "problem_statement": {
                "customer": [
                    "Can you confirm whether {transaction} was successful?",
                    "I would like to check the status of {transaction}.",
                    "Did my recent payment go through successfully?"
                ]
            },
            "acknowledge": {
                "agent": [
                    "Certainly. I can check the transaction status.",
                    "I can help confirm whether the payment was successful."
                ]
            },
            "verification_request": {
                "agent": [
                    "Please provide the approximate transaction details.",
                    "Could you confirm when the payment was made?"
                ]
            },
            "verification_response": {
                "customer": [
                    "It was made recently.",
                    "The transaction happened earlier today."
                ]
            },
            "inform": {
                "agent": [
                    "The transaction status has been confirmed successfully.",
                    "I have checked the payment record and the transaction information is available."
                ]
            },
            "confirm": {
                "customer": [
                    "Great, thank you.",
                    "Thank you for confirming."
                ]
            },
            "closing": {
                "agent": [
                    "You are welcome. Is there anything else I can help you with?",
                    "Happy to help. Have a good day."
                ]
            }
        }
    },

    # ========================================================
    # REFUNDS
    # ========================================================

    "REFUND_REQUEST": {
        "family": "Refunds",

        "entities": {
            "purchase": [
                "an online purchase",
                "a recent order",
                "a cancelled purchase"
            ],
            "refund_timeline": [
                "three to five working days",
                "five to seven working days",
                "up to ten business days"
            ]
        },

        "flow": [
            "problem_statement",
            "acknowledge",
            "verification_request",
            "verification_response",
            "resolve",
            "inform",
            "confirm",
            "closing"
        ],

        "templates": {
            "problem_statement": {
                "customer": [
                    "I would like to request a refund for {purchase}.",
                    "Can you help me request a refund for {purchase}?",
                    "I need assistance with getting a refund."
                ]
            },
            "acknowledge": {
                "agent": [
                    "Certainly. I can help you with the refund request.",
                    "I understand. Let us review the refund options."
                ]
            },
            "verification_request": {
                "agent": [
                    "Could you provide the transaction or order details?",
                    "Please confirm the reference associated with the purchase."
                ]
            },
            "verification_response": {
                "customer": [
                    "The transaction was made recently.",
                    "Yes, I have the purchase details available."
                ]
            },
            "resolve": {
                "agent": [
                    "Your refund request has been submitted successfully.",
                    "I have initiated the refund request."
                ]
            },
            "inform": {
                "agent": [
                    "The refund may take {refund_timeline} to be completed.",
                    "You can expect the refund within {refund_timeline}."
                ]
            },
            "confirm": {
                "customer": [
                    "Okay, thank you.",
                    "Thank you for your help."
                ]
            },
            "closing": {
                "agent": [
                    "You are welcome. Please contact us if you need further assistance.",
                    "Happy to help. Have a good day."
                ]
            }
        }
    },

    "REFUND_PENDING": {
        "family": "Refunds",

        "entities": {
            "refund_time": [
                "a few days ago",
                "last week",
                "earlier this month"
            ]
        },

        "flow": [
            "problem_statement",
            "acknowledge",
            "verification_request",
            "verification_response",
            "status_check",
            "inform",
            "confirm",
            "closing"
        ],

        "templates": {
            "problem_statement": {
                "customer": [
                    "My refund is still pending.",
                    "I requested a refund but it has not been completed yet.",
                    "Can you check the status of my pending refund?"
                ]
            },
            "acknowledge": {
                "agent": [
                    "Certainly. I can check the refund status.",
                    "I understand your concern. Let me review the refund."
                ]
            },
            "verification_request": {
                "agent": [
                    "Could you tell me when the refund was requested?",
                    "Please provide the approximate refund request date."
                ]
            },
            "verification_response": {
                "customer": [
                    "I requested it {refund_time}.",
                    "The refund request was submitted {refund_time}."
                ]
            },
            "status_check": {
                "agent": [
                    "I have checked the refund and it is currently being processed.",
                    "The refund is still in the processing stage."
                ]
            },
            "inform": {
                "agent": [
                    "Please allow additional processing time for the refund to be completed.",
                    "The refund status will update once processing is complete."
                ]
            },
            "confirm": {
                "customer": [
                    "Okay, thank you for checking.",
                    "Understood. I will wait."
                ]
            },
            "closing": {
                "agent": [
                    "You are welcome. Please contact us again if the refund remains pending.",
                    "Happy to help. Have a good day."
                ]
            }
        }
    },

    "REFUND_NOT_RECEIVED": {
        "family": "Refunds",

        "entities": {
            "expected_time": [
                "last week",
                "several days ago",
                "earlier this month"
            ]
        },

        "flow": [
            "problem_statement",
            "acknowledge",
            "verification_request",
            "verification_response",
            "status_check",
            "resolve",
            "inform",
            "closing"
        ],

        "templates": {
            "problem_statement": {
                "customer": [
                    "My refund was expected {expected_time}, but I have not received it.",
                    "I was told my refund was processed, but the money is not in my account.",
                    "Can you check why my refund has not arrived?"
                ]
            },
            "acknowledge": {
                "agent": [
                    "I understand your concern. Let me check the refund status.",
                    "I am sorry for the delay. I will review the refund information."
                ]
            },
            "verification_request": {
                "agent": [
                    "Please provide the relevant transaction information.",
                    "Could you confirm the refund reference details?"
                ]
            },
            "verification_response": {
                "customer": [
                    "Sure, I can provide the details.",
                    "Yes, I have the transaction information."
                ]
            },
            "status_check": {
                "agent": [
                    "I have reviewed the refund record.",
                    "Let me check the current processing status."
                ]
            },
            "resolve": {
                "agent": [
                    "I have raised the issue for further investigation.",
                    "A follow-up request has been created for the missing refund."
                ]
            },
            "inform": {
                "agent": [
                    "You will receive an update once the review is completed.",
                    "The relevant team will investigate and provide an update."
                ]
            },
            "closing": {
                "agent": [
                    "Thank you for your patience.",
                    "Please contact us again if you need further assistance."
                ]
            }
        }
    },

    # ========================================================
    # FRAUD AND DISPUTES
    # ========================================================

    "FRAUD_DISPUTE": {
        "family": "Fraud",

        "entities": {
            "transaction": [
                "a transaction I do not recognize",
                "an unfamiliar charge",
                "a suspicious payment"
            ]
        },

        "flow": [
            "problem_statement",
            "acknowledge",
            "transaction_verification",
            "customer_response",
            "security_action",
            "resolve",
            "inform",
            "confirm",
            "closing"
        ],

        "templates": {
            "problem_statement": {
                "customer": [
                    "I noticed {transaction} on my account.",
                    "There is {transaction} and I want to dispute it.",
                    "I believe there is {transaction} on my account."
                ]
            },
            "acknowledge": {
                "agent": [
                    "I understand your concern. I will help you review and dispute the transaction.",
                    "I am sorry to hear that. Let us investigate the transaction."
                ]
            },
            "transaction_verification": {
                "agent": [
                    "Can you confirm that you did not make or authorize this transaction?",
                    "Please confirm that the transaction is not recognized by you."
                ]
            },
            "customer_response": {
                "customer": [
                    "Yes, I did not make this transaction.",
                    "That is correct. I do not recognize the charge."
                ]
            },
            "security_action": {
                "agent": [
                    "I have taken the necessary security action to protect the account.",
                    "The account has been secured while the transaction is reviewed."
                ]
            },
            "resolve": {
                "agent": [
                    "I have registered a dispute for the transaction.",
                    "The dispute has now been submitted for investigation."
                ]
            },
            "inform": {
                "agent": [
                    "You will receive updates as the investigation progresses.",
                    "We will notify you once there is an update on the dispute."
                ]
            },
            "confirm": {
                "customer": [
                    "Thank you for helping me report it.",
                    "Okay, thank you for your help."
                ]
            },
            "closing": {
                "agent": [
                    "Please continue monitoring your account for unfamiliar activity.",
                    "Is there anything else I can help you with today?"
                ]
            }
        }
    },

    "MERCHANT_DISPUTE": {
        "family": "Disputes",

        "entities": {
            "issue": [
                "the merchant charged the wrong amount",
                "I did not receive the product or service",
                "the merchant did not resolve my complaint"
            ]
        },

        "flow": [
            "problem_statement",
            "acknowledge",
            "verification_request",
            "verification_response",
            "resolve",
            "inform",
            "confirm",
            "closing"
        ],

        "templates": {
            "problem_statement": {
                "customer": [
                    "I need help because {issue}.",
                    "I would like to dispute a merchant transaction because {issue}.",
                    "There is a problem with a merchant transaction. {issue}."
                ]
            },
            "acknowledge": {
                "agent": [
                    "I understand. I can help you begin the dispute process.",
                    "Let us review the merchant transaction."
                ]
            },
            "verification_request": {
                "agent": [
                    "Could you provide the transaction details?",
                    "Please confirm the relevant transaction information."
                ]
            },
            "verification_response": {
                "customer": [
                    "Sure, I can provide the transaction details.",
                    "Yes, I have the information available."
                ]
            },
            "resolve": {
                "agent": [
                    "The merchant dispute has been registered successfully.",
                    "I have submitted the dispute for review."
                ]
            },
            "inform": {
                "agent": [
                    "You will receive updates regarding the dispute review.",
                    "The review team will notify you when there is an update."
                ]
            },
            "confirm": {
                "customer": [
                    "Thank you for your help.",
                    "Okay, I understand."
                ]
            },
            "closing": {
                "agent": [
                    "You are welcome. Please contact us if you have further questions.",
                    "Happy to help. Have a good day."
                ]
            }
        }
    },

    "CHARGEBACK_REQUEST": {
        "family": "Disputes",

        "entities": {
            "reason": [
                "the purchase was not delivered",
                "the service was not provided",
                "the merchant refused to resolve the issue"
            ]
        },

        "flow": [
            "problem_statement",
            "acknowledge",
            "verification_request",
            "verification_response",
            "eligibility_check",
            "resolve",
            "inform",
            "closing"
        ],

        "templates": {
            "problem_statement": {
                "customer": [
                    "I would like to request a chargeback because {reason}.",
                    "Can you help me start a chargeback? {reason}.",
                    "I need to dispute a payment because {reason}."
                ]
            },
            "acknowledge": {
                "agent": [
                    "Certainly. I can help you review the chargeback request.",
                    "I understand. Let us check the available dispute options."
                ]
            },
            "verification_request": {
                "agent": [
                    "Please provide the transaction information.",
                    "Could you confirm the relevant purchase details?"
                ]
            },
            "verification_response": {
                "customer": [
                    "Yes, I have the transaction details.",
                    "Sure, I can provide the information."
                ]
            },
            "eligibility_check": {
                "agent": [
                    "I have reviewed the transaction for chargeback eligibility.",
                    "Let me check whether the transaction meets the required criteria."
                ]
            },
            "resolve": {
                "agent": [
                    "Your chargeback request has been submitted for review.",
                    "I have initiated the chargeback process."
                ]
            },
            "inform": {
                "agent": [
                    "You will receive updates as the case progresses.",
                    "The dispute team will notify you about the next steps."
                ]
            },
            "closing": {
                "agent": [
                    "Thank you for providing the information.",
                    "Please contact us again if you need further assistance."
                ]
            }
        }
    },

    # ========================================================
    # AUTHENTICATION
    # ========================================================

    "LOGIN_ISSUE": {
        "family": "Authentication",

        "entities": {
            "channel": [
                "mobile app",
                "website"
            ]
        },

        "flow": [
            "problem_statement",
            "acknowledge",
            "probing_question",
            "customer_response",
            "troubleshooting",
            "resolve",
            "confirm",
            "closing"
        ],

        "templates": {
            "problem_statement": {
                "customer": [
                    "I cannot sign in to my account.",
                    "My login keeps failing.",
                    "I need help because I cannot access my account."
                ]
            },
            "acknowledge": {
                "agent": [
                    "I understand. I will help you regain access.",
                    "I am sorry you are having trouble signing in."
                ]
            },
            "probing_question": {
                "agent": [
                    "Are you trying to sign in through the {channel}?",
                    "Could you tell me whether you are using the mobile app or website?"
                ]
            },
            "customer_response": {
                "customer": [
                    "Yes, I am using the {channel}.",
                    "I have been trying through the {channel}."
                ]
            },
            "troubleshooting": {
                "agent": [
                    "I have checked the account access status.",
                    "Let me review whether there is an access restriction."
                ]
            },
            "resolve": {
                "agent": [
                    "The access issue has been addressed. Please try signing in again.",
                    "I have completed the necessary access check. You can try again."
                ]
            },
            "confirm": {
                "customer": [
                    "Okay, I will try again.",
                    "Thank you. I will test it now."
                ]
            },
            "closing": {
                "agent": [
                    "You are welcome. Please contact us again if the issue continues.",
                    "Happy to help. Have a good day."
                ]
            }
        }
    },

    "PASSWORD_RESET": {
        "family": "Authentication",

        "entities": {
            "channel": [
                "mobile app",
                "website"
            ]
        },

        "flow": [
            "problem_statement",
            "acknowledge",
            "verification_request",
            "verification_response",
            "resolve",
            "inform",
            "confirm",
            "closing"
        ],

        "templates": {
            "problem_statement": {
                "customer": [
                    "I forgot my password and need to reset it.",
                    "I need help resetting my account password.",
                    "I cannot remember my password."
                ]
            },
            "acknowledge": {
                "agent": [
                    "Certainly. I can help you reset your password.",
                    "I understand. Let us complete the password reset process."
                ]
            },
            "verification_request": {
                "agent": [
                    "For security, please complete account verification.",
                    "Before resetting the password, I need to verify your identity."
                ]
            },
            "verification_response": {
                "customer": [
                    "Sure, I can complete the verification.",
                    "Yes, please proceed."
                ]
            },
            "resolve": {
                "agent": [
                    "The password reset has been initiated successfully.",
                    "I have completed the password reset request."
                ]
            },
            "inform": {
                "agent": [
                    "Please follow the password reset instructions through the {channel}.",
                    "You can now create a new password using the reset process."
                ]
            },
            "confirm": {
                "customer": [
                    "Thank you. I will do that.",
                    "Okay, thank you for your help."
                ]
            },
            "closing": {
                "agent": [
                    "You are welcome. Is there anything else I can help you with?",
                    "Happy to help. Have a good day."
                ]
            }
        }
    },

    "ACCOUNT_LOCKED": {
        "family": "Authentication",

        "entities": {
            "reason": [
                "too many unsuccessful login attempts",
                "an account security check",
                "a possible authentication issue"
            ]
        },

        "flow": [
            "problem_statement",
            "acknowledge",
            "verification_request",
            "verification_response",
            "status_check",
            "resolve",
            "confirm",
            "closing"
        ],

        "templates": {
            "problem_statement": {
                "customer": [
                    "My account is locked and I cannot sign in.",
                    "I believe my account was locked because of {reason}.",
                    "Can you help me unlock my account?"
                ]
            },
            "acknowledge": {
                "agent": [
                    "I understand. Let me help you review the account lock.",
                    "I can help you regain access after verification."
                ]
            },
            "verification_request": {
                "agent": [
                    "For security, please complete identity verification.",
                    "Before unlocking the account, I need to verify your identity."
                ]
            },
            "verification_response": {
                "customer": [
                    "Sure, I can verify my details.",
                    "Yes, please proceed."
                ]
            },
            "status_check": {
                "agent": [
                    "I have reviewed the account status.",
                    "Let me check the current access restriction."
                ]
            },
            "resolve": {
                "agent": [
                    "The account access restriction has been removed.",
                    "Your account has been unlocked successfully."
                ]
            },
            "confirm": {
                "customer": [
                    "Great, thank you.",
                    "Thank you for helping me regain access."
                ]
            },
            "closing": {
                "agent": [
                    "You are welcome. Please contact us if you continue to experience issues.",
                    "Happy to help. Have a good day."
                ]
            }
        }
    },

    "OTP_NOT_RECEIVED": {
        "family": "Authentication",

        "entities": {
            "channel": [
                "mobile phone",
                "registered email",
                "authentication application"
            ]
        },

        "flow": [
            "problem_statement",
            "acknowledge",
            "probing_question",
            "customer_response",
            "troubleshooting",
            "resolve",
            "confirm",
            "closing"
        ],

        "templates": {
            "problem_statement": {
                "customer": [
                    "I am not receiving my one-time password.",
                    "The verification code has not arrived.",
                    "I need help because I did not receive the OTP."
                ]
            },
            "acknowledge": {
                "agent": [
                    "I understand. Let us check the verification delivery issue.",
                    "Certainly. I can help troubleshoot the OTP problem."
                ]
            },
            "probing_question": {
                "agent": [
                    "Are you expecting the code through your {channel}?",
                    "Could you confirm where the verification code should be delivered?"
                ]
            },
            "customer_response": {
                "customer": [
                    "Yes, I am expecting it on my {channel}.",
                    "The code should arrive through my {channel}."
                ]
            },
            "troubleshooting": {
                "agent": [
                    "I have checked the verification delivery status.",
                    "Let me review the authentication delivery information."
                ]
            },
            "resolve": {
                "agent": [
                    "I have initiated a new verification code request.",
                    "A new OTP has been requested for your account."
                ]
            },
            "confirm": {
                "customer": [
                    "Okay, I will check again.",
                    "Thank you for your help."
                ]
            },
            "closing": {
                "agent": [
                    "Please contact us again if you still do not receive the code.",
                    "Happy to help. Have a good day."
                ]
            }
        }
    },

    "BIOMETRIC_LOGIN_ISSUE": {
        "family": "Authentication",

        "entities": {
            "biometric_type": [
                "fingerprint login",
                "face recognition",
                "biometric authentication"
            ]
        },

        "flow": [
            "problem_statement",
            "acknowledge",
            "probing_question",
            "customer_response",
            "troubleshooting",
            "inform",
            "confirm",
            "closing"
        ],

        "templates": {
            "problem_statement": {
                "customer": [
                    "My {biometric_type} is not working.",
                    "I cannot log in using {biometric_type}.",
                    "I am having trouble with {biometric_type}."
                ]
            },
            "acknowledge": {
                "agent": [
                    "I understand. Let us troubleshoot the biometric login issue.",
                    "Certainly. I can help you review the authentication problem."
                ]
            },
            "probing_question": {
                "agent": [
                    "Are you seeing an error when you attempt to use {biometric_type}?",
                    "Could you tell me whether the biometric option fails every time?"
                ]
            },
            "customer_response": {
                "customer": [
                    "Yes, it fails whenever I try to use it.",
                    "The biometric login option is not completing successfully."
                ]
            },
            "troubleshooting": {
                "agent": [
                    "Please ensure the application is updated and biometric access is enabled.",
                    "I recommend refreshing the biometric settings in your device and application."
                ]
            },
            "inform": {
                "agent": [
                    "You can temporarily use your password while the biometric issue is resolved.",
                    "Password login can be used as an alternative authentication method."
                ]
            },
            "confirm": {
                "customer": [
                    "Okay, I will try that.",
                    "Thank you for the information."
                ]
            },
            "closing": {
                "agent": [
                    "You are welcome. Please contact us if the issue continues.",
                    "Happy to help. Have a good day."
                ]
            }
        }
    },

    # ========================================================
    # ACCOUNT INFORMATION AND MANAGEMENT
    # ========================================================

    "ACCOUNT_BALANCE_QUERY": {
        "family": "Account Information",

        "entities": {
            "balance_type": [
                "available balance",
                "current balance",
                "available credit"
            ]
        },

        "flow": [
            "problem_statement",
            "acknowledge",
            "verification_request",
            "verification_response",
            "inform",
            "confirm",
            "closing"
        ],

        "templates": {
            "problem_statement": {
                "customer": [
                    "I would like to check my {balance_type}.",
                    "Can you tell me my {balance_type}?",
                    "I need information about my account balance."
                ]
            },
            "acknowledge": {
                "agent": [
                    "Certainly. I can help you check that information.",
                    "I can look up your account balance."
                ]
            },
            "verification_request": {
                "agent": [
                    "For security, please complete account verification.",
                    "Before accessing account information, I need to verify your identity."
                ]
            },
            "verification_response": {
                "customer": [
                    "Sure, please proceed.",
                    "Yes, I can verify my details."
                ]
            },
            "inform": {
                "agent": [
                    "The requested {balance_type} information is now available.",
                    "I have retrieved your current {balance_type} details."
                ]
            },
            "confirm": {
                "customer": [
                    "Thank you, that is helpful.",
                    "Okay, that answers my question."
                ]
            },
            "closing": {
                "agent": [
                    "You are welcome. Is there anything else I can help you with?",
                    "Happy to help. Have a good day."
                ]
            }
        }
    },

    "ACCOUNT_STATEMENT": {
        "family": "Account Information",

        "entities": {
            "statement_period": [
                "the latest month",
                "the previous month",
                "the last three months"
            ],
            "delivery_channel": [
                "registered email",
                "online account",
                "mobile application"
            ]
        },

        "flow": [
            "problem_statement",
            "acknowledge",
            "verification_request",
            "verification_response",
            "resolve",
            "inform",
            "confirm",
            "closing"
        ],

        "templates": {
            "problem_statement": {
                "customer": [
                    "I need an account statement for {statement_period}.",
                    "Can you help me get my statement for {statement_period}?",
                    "I would like to request an account statement."
                ]
            },
            "acknowledge": {
                "agent": [
                    "Certainly. I can help you request the statement.",
                    "I can assist you with getting a copy of your statement."
                ]
            },
            "verification_request": {
                "agent": [
                    "For security, please complete account verification.",
                    "Before I process the request, I need to verify your account."
                ]
            },
            "verification_response": {
                "customer": [
                    "Sure, I can verify my details.",
                    "Yes, please proceed."
                ]
            },
            "resolve": {
                "agent": [
                    "The statement request has been processed successfully.",
                    "I have generated the requested statement."
                ]
            },
            "inform": {
                "agent": [
                    "The statement will be available through your {delivery_channel}.",
                    "You can access the statement through your {delivery_channel}."
                ]
            },
            "confirm": {
                "customer": [
                    "Thank you for your help.",
                    "Great, thank you."
                ]
            },
            "closing": {
                "agent": [
                    "You are welcome. Is there anything else I can help you with?",
                    "Happy to help. Have a good day."
                ]
            }
        }
    },

    "ACCOUNT_CLOSURE": {
        "family": "Account Management",

        "entities": {
            "reason": [
                "I no longer need the account",
                "I am switching to another service",
                "I want to simplify my finances"
            ]
        },

        "flow": [
            "problem_statement",
            "acknowledge",
            "verification_request",
            "verification_response",
            "eligibility_check",
            "resolve",
            "inform",
            "closing"
        ],

        "templates": {
            "problem_statement": {
                "customer": [
                    "I would like to close my account because {reason}.",
                    "Can you help me close my account?",
                    "I want to permanently close my account."
                ]
            },
            "acknowledge": {
                "agent": [
                    "I understand. I can explain the account closure process.",
                    "Certainly. Let us review the requirements before closing the account."
                ]
            },
            "verification_request": {
                "agent": [
                    "For security, please complete identity verification.",
                    "Before processing the request, I need to verify your account."
                ]
            },
            "verification_response": {
                "customer": [
                    "Sure, please proceed.",
                    "Yes, I can complete the verification."
                ]
            },
            "eligibility_check": {
                "agent": [
                    "I have reviewed the account for any pending requirements.",
                    "Let me check whether there are any outstanding balances or restrictions."
                ]
            },
            "resolve": {
                "agent": [
                    "Your account closure request has been submitted.",
                    "I have initiated the account closure process."
                ]
            },
            "inform": {
                "agent": [
                    "You will receive confirmation once the closure process is completed.",
                    "We will notify you when the account closure has been finalized."
                ]
            },
            "closing": {
                "agent": [
                    "Thank you for contacting us.",
                    "Please contact us again if you have any questions."
                ]
            }
        }
    },

    "ACCOUNT_REOPENING": {
        "family": "Account Management",

        "entities": {
            "closure_time": [
                "recently",
                "a few weeks ago",
                "earlier this month"
            ]
        },

        "flow": [
            "problem_statement",
            "acknowledge",
            "verification_request",
            "verification_response",
            "eligibility_check",
            "inform",
            "confirm",
            "closing"
        ],

        "templates": {
            "problem_statement": {
                "customer": [
                    "I closed my account {closure_time} and would like to reopen it.",
                    "Can you tell me if my previously closed account can be reopened?",
                    "I would like to restore my old account."
                ]
            },
            "acknowledge": {
                "agent": [
                    "Certainly. I can check the available account reopening options.",
                    "I understand. Let me review the account status."
                ]
            },
            "verification_request": {
                "agent": [
                    "Please complete identity verification before I access the account information.",
                    "For security, I need to verify your identity."
                ]
            },
            "verification_response": {
                "customer": [
                    "Sure, please proceed.",
                    "Yes, I can verify my details."
                ]
            },
            "eligibility_check": {
                "agent": [
                    "I have reviewed whether the account is eligible for reopening.",
                    "Let me check the account status and reopening options."
                ]
            },
            "inform": {
                "agent": [
                    "The available next steps for reopening the account have been identified.",
                    "You will be guided through the applicable account reopening process."
                ]
            },
            "confirm": {
                "customer": [
                    "Okay, thank you for explaining.",
                    "Thank you. I understand the next steps."
                ]
            },
            "closing": {
                "agent": [
                    "You are welcome. Please contact us if you need further assistance.",
                    "Happy to help. Have a good day."
                ]
            }
        }
    },

    "ADDRESS_CHANGE": {
        "family": "Profile Management",

        "entities": {
            "address_type": [
                "home address",
                "mailing address",
                "registered address"
            ]
        },

        "flow": [
            "problem_statement",
            "acknowledge",
            "verification_request",
            "verification_response",
            "information_request",
            "customer_response",
            "resolve",
            "confirm",
            "closing"
        ],

        "templates": {
            "problem_statement": {
                "customer": [
                    "I need to update my {address_type}.",
                    "I recently moved and want to change my {address_type}.",
                    "Can you help me update my address?"
                ]
            },
            "acknowledge": {
                "agent": [
                    "Certainly. I can help you update your address.",
                    "I understand. Let us update the address on your account."
                ]
            },
            "verification_request": {
                "agent": [
                    "Before making changes, I need to verify your identity.",
                    "For security, please complete account verification."
                ]
            },
            "verification_response": {
                "customer": [
                    "Sure, I can verify my details.",
                    "Yes, please proceed."
                ]
            },
            "information_request": {
                "agent": [
                    "Please provide the new address you would like to register.",
                    "What address should be updated on the account?"
                ]
            },
            "customer_response": {
                "customer": [
                    "I would like to update it to my new home address.",
                    "Please update the account with my new mailing address."
                ]
            },
            "resolve": {
                "agent": [
                    "The address update has been completed successfully.",
                    "I have updated the address on your account."
                ]
            },
            "confirm": {
                "customer": [
                    "Great, thank you.",
                    "Thank you for updating it."
                ]
            },
            "closing": {
                "agent": [
                    "You are welcome. Is there anything else I can help you with?",
                    "Happy to help. Have a good day."
                ]
            }
        }
    },

    "PHONE_NUMBER_CHANGE": {
        "family": "Profile Management",

        "entities": {
            "reason": [
                "I have a new mobile number",
                "my old number is no longer active",
                "I changed my phone provider"
            ]
        },

        "flow": [
            "problem_statement",
            "acknowledge",
            "verification_request",
            "verification_response",
            "information_request",
            "customer_response",
            "resolve",
            "confirm",
            "closing"
        ],

        "templates": {
            "problem_statement": {
                "customer": [
                    "I need to change my phone number because {reason}.",
                    "Can you help me update my registered mobile number?",
                    "I want to replace the phone number on my account."
                ]
            },
            "acknowledge": {
                "agent": [
                    "Certainly. I can help you update your phone number.",
                    "I understand. Let us update the contact information."
                ]
            },
            "verification_request": {
                "agent": [
                    "For security, please complete identity verification.",
                    "Before changing the phone number, I need to verify your identity."
                ]
            },
            "verification_response": {
                "customer": [
                    "Sure, please proceed.",
                    "Yes, I can verify my details."
                ]
            },
            "information_request": {
                "agent": [
                    "Please provide the new phone number you want to register.",
                    "What new number should be associated with the account?"
                ]
            },
            "customer_response": {
                "customer": [
                    "I would like to register my new mobile number.",
                    "Please replace the old number with my new one."
                ]
            },
            "resolve": {
                "agent": [
                    "Your registered phone number has been updated.",
                    "I have successfully processed the phone number change."
                ]
            },
            "confirm": {
                "customer": [
                    "Great, thank you.",
                    "Thank you for your help."
                ]
            },
            "closing": {
                "agent": [
                    "You are welcome. Please contact us if you need further assistance.",
                    "Happy to help. Have a good day."
                ]
            }
        }
    },

    "EMAIL_CHANGE": {
        "family": "Profile Management",

        "entities": {
            "reason": [
                "I have a new email address",
                "I no longer use my old email",
                "my previous email account is inaccessible"
            ]
        },

        "flow": [
            "problem_statement",
            "acknowledge",
            "verification_request",
            "verification_response",
            "information_request",
            "customer_response",
            "resolve",
            "confirm",
            "closing"
        ],

        "templates": {
            "problem_statement": {
                "customer": [
                    "I need to change my registered email because {reason}.",
                    "Can you help me update the email address on my account?",
                    "I want to replace my current email address."
                ]
            },
            "acknowledge": {
                "agent": [
                    "Certainly. I can help you update your email address.",
                    "I understand. Let us update your contact information."
                ]
            },
            "verification_request": {
                "agent": [
                    "For security, please complete identity verification.",
                    "Before updating the email, I need to verify your identity."
                ]
            },
            "verification_response": {
                "customer": [
                    "Sure, please proceed.",
                    "Yes, I can verify my details."
                ]
            },
            "information_request": {
                "agent": [
                    "Please provide the new email address you would like to register.",
                    "What email address should be associated with the account?"
                ]
            },
            "customer_response": {
                "customer": [
                    "I would like to register my new email address.",
                    "Please replace my old email with the new one."
                ]
            },
            "resolve": {
                "agent": [
                    "Your registered email address has been updated.",
                    "I have successfully processed the email change."
                ]
            },
            "confirm": {
                "customer": [
                    "Thank you for your help.",
                    "Great, thank you."
                ]
            },
            "closing": {
                "agent": [
                    "You are welcome. Please contact us if you need further assistance.",
                    "Happy to help. Have a good day."
                ]
            }
        }
    },

    # ========================================================
    # TRANSFERS
    # ========================================================

    "BANK_TRANSFER_PENDING": {
        "family": "Transfers",

        "entities": {
            "transfer_time": [
                "a few minutes ago",
                "earlier today",
                "yesterday"
            ]
        },

        "flow": [
            "problem_statement",
            "acknowledge",
            "verification_request",
            "verification_response",
            "status_check",
            "inform",
            "confirm",
            "closing"
        ],

        "templates": {
            "problem_statement": {
                "customer": [
                    "My bank transfer is still pending.",
                    "I made a transfer but the status has not updated.",
                    "Can you check why my transfer has not completed?"
                ]
            },
            "acknowledge": {
                "agent": [
                    "Certainly. I can check the transfer status.",
                    "I understand your concern. Let me review the transfer."
                ]
            },
            "verification_request": {
                "agent": [
                    "Could you confirm when the transfer was initiated?",
                    "Please provide the approximate transfer time."
                ]
            },
            "verification_response": {
                "customer": [
                    "I made the transfer {transfer_time}.",
                    "The transfer was initiated {transfer_time}."
                ]
            },
            "status_check": {
                "agent": [
                    "I have checked the transfer and it is still being processed.",
                    "The transfer is currently pending in the processing system."
                ]
            },
            "inform": {
                "agent": [
                    "Please allow additional processing time for the transfer to complete.",
                    "The transfer status should update automatically once processing is finished."
                ]
            },
            "confirm": {
                "customer": [
                    "Okay, thank you for checking.",
                    "Understood. I will wait."
                ]
            },
            "closing": {
                "agent": [
                    "You are welcome. Please contact us again if the transfer remains pending.",
                    "Happy to help. Have a good day."
                ]
            }
        }
    },

    "BANK_TRANSFER_FAILED": {
        "family": "Transfers",

        "entities": {
            "transfer_type": [
                "a bank transfer",
                "a domestic transfer",
                "an account-to-account transfer"
            ]
        },

        "flow": [
            "problem_statement",
            "acknowledge",
            "probing_question",
            "customer_response",
            "status_check",
            "resolve",
            "confirm",
            "closing"
        ],

        "templates": {
            "problem_statement": {
                "customer": [
                    "My {transfer_type} failed.",
                    "I tried to make {transfer_type}, but it did not go through.",
                    "Can you help me because my transfer was unsuccessful?"
                ]
            },
            "acknowledge": {
                "agent": [
                    "I understand. Let me help you check the transfer issue.",
                    "I am sorry you experienced this problem."
                ]
            },
            "probing_question": {
                "agent": [
                    "Did you receive an error message when attempting the transfer?",
                    "Could you tell me whether the transfer failed immediately or after processing?"
                ]
            },
            "customer_response": {
                "customer": [
                    "Yes, I received an error during the transfer.",
                    "The transfer did not complete successfully."
                ]
            },
            "status_check": {
                "agent": [
                    "I have reviewed the transfer status.",
                    "Let me check the transaction record."
                ]
            },
            "resolve": {
                "agent": [
                    "The transfer issue has been reviewed. You can try the transfer again.",
                    "The necessary checks have been completed and the transfer can be retried."
                ]
            },
            "confirm": {
                "customer": [
                    "Okay, I will try again.",
                    "Thank you for checking."
                ]
            },
            "closing": {
                "agent": [
                    "You are welcome. Please contact us again if the issue continues.",
                    "Happy to help. Have a good day."
                ]
            }
        }
    },

    "TRANSFER_CANCELLATION": {
        "family": "Transfers",

        "entities": {
            "transfer_status": [
                "still pending",
                "not yet completed",
                "currently processing"
            ]
        },

        "flow": [
            "problem_statement",
            "acknowledge",
            "verification_request",
            "verification_response",
            "eligibility_check",
            "resolve",
            "inform",
            "closing"
        ],

        "templates": {
            "problem_statement": {
                "customer": [
                    "I want to cancel a transfer that is {transfer_status}.",
                    "Can you help me cancel my bank transfer?",
                    "I sent a transfer by mistake and want to stop it."
                ]
            },
            "acknowledge": {
                "agent": [
                    "Certainly. I can check whether the transfer can still be cancelled.",
                    "I understand. Let us review the transfer status."
                ]
            },
            "verification_request": {
                "agent": [
                    "Please provide the relevant transfer details.",
                    "For security, I need to verify your account."
                ]
            },
            "verification_response": {
                "customer": [
                    "Sure, please proceed.",
                    "Yes, I can provide the details."
                ]
            },
            "eligibility_check": {
                "agent": [
                    "I have checked whether the transfer is eligible for cancellation.",
                    "Let me review the current processing status."
                ]
            },
            "resolve": {
                "agent": [
                    "The transfer cancellation request has been submitted.",
                    "I have initiated the transfer cancellation process."
                ]
            },
            "inform": {
                "agent": [
                    "You will receive an update regarding the cancellation outcome.",
                    "We will notify you once the cancellation request is processed."
                ]
            },
            "closing": {
                "agent": [
                    "Thank you for contacting us.",
                    "Please contact us again if you need further assistance."
                ]
            }
        }
    },

    "BENEFICIARY_ADD": {
        "family": "Transfers",

        "entities": {
            "beneficiary_type": [
                "a family member",
                "a personal bank account",
                "a new payee"
            ]
        },

        "flow": [
            "problem_statement",
            "acknowledge",
            "verification_request",
            "verification_response",
            "information_request",
            "customer_response",
            "resolve",
            "confirm",
            "closing"
        ],

        "templates": {
            "problem_statement": {
                "customer": [
                    "I would like to add {beneficiary_type} as a beneficiary.",
                    "Can you help me add a new beneficiary?",
                    "I need assistance adding a new payee."
                ]
            },
            "acknowledge": {
                "agent": [
                    "Certainly. I can guide you through adding the beneficiary.",
                    "I understand. Let us proceed with the beneficiary setup."
                ]
            },
            "verification_request": {
                "agent": [
                    "For security, please complete account verification.",
                    "Before making changes, I need to verify your identity."
                ]
            },
            "verification_response": {
                "customer": [
                    "Sure, please proceed.",
                    "Yes, I can complete verification."
                ]
            },
            "information_request": {
                "agent": [
                    "Please provide the required beneficiary details.",
                    "What beneficiary information would you like to register?"
                ]
            },
            "customer_response": {
                "customer": [
                    "I have the beneficiary details ready.",
                    "Sure, I can provide the required information."
                ]
            },
            "resolve": {
                "agent": [
                    "The beneficiary has been added successfully.",
                    "I have completed the beneficiary registration request."
                ]
            },
            "confirm": {
                "customer": [
                    "Great, thank you.",
                    "Thank you for your help."
                ]
            },
            "closing": {
                "agent": [
                    "You are welcome. Please contact us if you need further assistance.",
                    "Happy to help. Have a good day."
                ]
            }
        }
    },

    "BENEFICIARY_REMOVE": {
        "family": "Transfers",

        "entities": {
            "beneficiary_type": [
                "an old payee",
                "a beneficiary I no longer use",
                "an outdated bank account"
            ]
        },

        "flow": [
            "problem_statement",
            "acknowledge",
            "verification_request",
            "verification_response",
            "information_request",
            "customer_response",
            "resolve",
            "confirm",
            "closing"
        ],

        "templates": {
            "problem_statement": {
                "customer": [
                    "I would like to remove {beneficiary_type}.",
                    "Can you help me delete a beneficiary?",
                    "I need to remove a payee from my account."
                ]
            },
            "acknowledge": {
                "agent": [
                    "Certainly. I can help you remove the beneficiary.",
                    "I understand. Let us review the beneficiary details."
                ]
            },
            "verification_request": {
                "agent": [
                    "For security, please complete account verification.",
                    "Before making changes, I need to verify your identity."
                ]
            },
            "verification_response": {
                "customer": [
                    "Sure, please proceed.",
                    "Yes, I can verify my details."
                ]
            },
            "information_request": {
                "agent": [
                    "Please identify the beneficiary you want to remove.",
                    "Could you provide the relevant beneficiary details?"
                ]
            },
            "customer_response": {
                "customer": [
                    "I would like to remove the selected beneficiary.",
                    "Yes, I can identify the beneficiary."
                ]
            },
            "resolve": {
                "agent": [
                    "The beneficiary has been removed successfully.",
                    "I have completed the beneficiary removal request."
                ]
            },
            "confirm": {
                "customer": [
                    "Great, thank you.",
                    "Thank you for your help."
                ]
            },
            "closing": {
                "agent": [
                    "You are welcome. Is there anything else I can help you with?",
                    "Happy to help. Have a good day."
                ]
            }
        }
    },

    # ========================================================
    # REWARDS
    # ========================================================

    "CASHBACK_MISSING": {
        "family": "Rewards",

        "entities": {
            "purchase": [
                "a recent eligible purchase",
                "an online transaction",
                "a promotional purchase"
            ]
        },

        "flow": [
            "problem_statement",
            "acknowledge",
            "verification_request",
            "verification_response",
            "status_check",
            "resolve",
            "inform",
            "closing"
        ],

        "templates": {
            "problem_statement": {
                "customer": [
                    "I did not receive cashback for {purchase}.",
                    "My cashback is missing after a recent transaction.",
                    "Can you check why I have not received my cashback?"
                ]
            },
            "acknowledge": {
                "agent": [
                    "Certainly. I can help you review the cashback status.",
                    "I understand your concern. Let me check the reward information."
                ]
            },
            "verification_request": {
                "agent": [
                    "Could you provide the approximate transaction details?",
                    "Please confirm the relevant purchase information."
                ]
            },
            "verification_response": {
                "customer": [
                    "Sure, I can provide the transaction details.",
                    "Yes, the purchase information is available."
                ]
            },
            "status_check": {
                "agent": [
                    "I have reviewed the cashback eligibility information.",
                    "Let me check the reward processing status."
                ]
            },
            "resolve": {
                "agent": [
                    "The missing cashback issue has been registered for review.",
                    "I have initiated a review of the cashback transaction."
                ]
            },
            "inform": {
                "agent": [
                    "You will receive an update once the review is completed.",
                    "The reward status will be updated after processing."
                ]
            },
            "closing": {
                "agent": [
                    "Thank you for contacting us.",
                    "Please contact us again if you need further assistance."
                ]
            }
        }
    },

    "REWARD_POINTS_QUERY": {
        "family": "Rewards",

        "entities": {
            "points_type": [
                "available reward points",
                "total reward points",
                "recently earned points"
            ]
        },

        "flow": [
            "problem_statement",
            "acknowledge",
            "verification_request",
            "verification_response",
            "inform",
            "confirm",
            "closing"
        ],

        "templates": {
            "problem_statement": {
                "customer": [
                    "I would like to know my {points_type}.",
                    "Can you check how many reward points I have?",
                    "I need information about my reward points."
                ]
            },
            "acknowledge": {
                "agent": [
                    "Certainly. I can check your reward point information.",
                    "I can help you review your available points."
                ]
            },
            "verification_request": {
                "agent": [
                    "For security, please complete account verification.",
                    "Before accessing reward information, I need to verify your identity."
                ]
            },
            "verification_response": {
                "customer": [
                    "Sure, please proceed.",
                    "Yes, I can verify my details."
                ]
            },
            "inform": {
                "agent": [
                    "The requested {points_type} information is now available.",
                    "I have retrieved your {points_type} details."
                ]
            },
            "confirm": {
                "customer": [
                    "Thank you, that is helpful.",
                    "Great, thank you."
                ]
            },
            "closing": {
                "agent": [
                    "You are welcome. Is there anything else I can help you with?",
                    "Happy to help. Have a good day."
                ]
            }
        }
    },

    "REWARD_REDEMPTION": {
        "family": "Rewards",

        "entities": {
            "redemption_type": [
                "travel rewards",
                "gift vouchers",
                "statement credit"
            ]
        },

        "flow": [
            "problem_statement",
            "acknowledge",
            "verification_request",
            "verification_response",
            "eligibility_check",
            "resolve",
            "confirm",
            "closing"
        ],

        "templates": {
            "problem_statement": {
                "customer": [
                    "I would like to redeem my points for {redemption_type}.",
                    "Can you help me redeem my reward points?",
                    "I want to use my reward points for {redemption_type}."
                ]
            },
            "acknowledge": {
                "agent": [
                    "Certainly. I can help you with the reward redemption.",
                    "I understand. Let us review the available redemption options."
                ]
            },
            "verification_request": {
                "agent": [
                    "Please complete account verification before proceeding.",
                    "For security, I need to verify your identity."
                ]
            },
            "verification_response": {
                "customer": [
                    "Sure, please proceed.",
                    "Yes, I can verify my details."
                ]
            },
            "eligibility_check": {
                "agent": [
                    "I have checked the available reward redemption options.",
                    "Let me review whether you have sufficient points."
                ]
            },
            "resolve": {
                "agent": [
                    "Your reward redemption request has been processed.",
                    "I have successfully submitted the redemption request."
                ]
            },
            "confirm": {
                "customer": [
                    "Great, thank you.",
                    "Thank you for your help."
                ]
            },
            "closing": {
                "agent": [
                    "You are welcome. Enjoy your rewards.",
                    "Happy to help. Have a good day."
                ]
            }
        }
    },

    # ========================================================
    # INTERNATIONAL SERVICES
    # ========================================================

    "FOREIGN_TRANSACTION_QUERY": {
        "family": "International Services",

        "entities": {
            "country": [
                "Singapore",
                "the United Kingdom",
                "the United States",
                "Japan"
            ]
        },

        "flow": [
            "problem_statement",
            "acknowledge",
            "verification_request",
            "verification_response",
            "inform",
            "confirm",
            "closing"
        ],

        "templates": {
            "problem_statement": {
                "customer": [
                    "Can I use my card while traveling to {country}?",
                    "I would like to know if foreign transactions are supported in {country}.",
                    "Do I need to do anything before using my card in {country}?"
                ]
            },
            "acknowledge": {
                "agent": [
                    "Certainly. I can provide information about international card usage.",
                    "I can help you understand the foreign transaction process."
                ]
            },
            "verification_request": {
                "agent": [
                    "For account-specific information, please complete verification.",
                    "Before reviewing account settings, I need to verify your identity."
                ]
            },
            "verification_response": {
                "customer": [
                    "Sure, please proceed.",
                    "Yes, I can verify my details."
                ]
            },
            "inform": {
                "agent": [
                    "Your account can be reviewed for international transaction availability.",
                    "Please ensure your account and card settings are appropriate before traveling."
                ]
            },
            "confirm": {
                "customer": [
                    "Thank you for the information.",
                    "Okay, that is helpful."
                ]
            },
            "closing": {
                "agent": [
                    "You are welcome. Have a safe trip.",
                    "Happy to help. Please contact us if you need further assistance."
                ]
            }
        }
    },

    "CURRENCY_CONVERSION_QUERY": {
        "family": "International Services",

        "entities": {
            "currency_pair": [
                "US dollars to Indian rupees",
                "British pounds to Indian rupees",
                "Singapore dollars to Indian rupees"
            ]
        },

        "flow": [
            "problem_statement",
            "acknowledge",
            "probing_question",
            "customer_response",
            "inform",
            "confirm",
            "closing"
        ],

        "templates": {
            "problem_statement": {
                "customer": [
                    "How does currency conversion work for foreign transactions?",
                    "Can you explain the exchange rate used for my card transaction?",
                    "I have a question about foreign currency conversion."
                ]
            },
            "acknowledge": {
                "agent": [
                    "Certainly. I can explain how currency conversion is applied.",
                    "I can help clarify the exchange rate process."
                ]
            },
            "probing_question": {
                "agent": [
                    "Which currency conversion are you asking about?",
                    "Could you tell me the currencies involved in the transaction?"
                ]
            },
            "customer_response": {
                "customer": [
                    "I would like to understand the conversion from {currency_pair}.",
                    "The transaction involves {currency_pair}."
                ]
            },
            "inform": {
                "agent": [
                    "The final converted amount can depend on the applicable exchange rate and transaction processing time.",
                    "Currency conversion is calculated using the applicable rate at the time defined by the transaction process."
                ]
            },
            "confirm": {
                "customer": [
                    "Thank you for explaining.",
                    "Okay, I understand."
                ]
            },
            "closing": {
                "agent": [
                    "You are welcome. Please contact us if you have more questions.",
                    "Happy to help. Have a good day."
                ]
            }
        }
    },

    # ========================================================
    # SUBSCRIPTIONS AND FEES
    # ========================================================

    "SUBSCRIPTION_CANCELLATION": {
        "family": "Subscriptions",

        "entities": {
            "subscription": [
                "a streaming subscription",
                "a software subscription",
                "a recurring service"
            ]
        },

        "flow": [
            "problem_statement",
            "acknowledge",
            "verification_request",
            "verification_response",
            "resolve",
            "inform",
            "confirm",
            "closing"
        ],

        "templates": {
            "problem_statement": {
                "customer": [
                    "I want to cancel {subscription}.",
                    "Can you help me stop a recurring subscription charge?",
                    "I need assistance cancelling a subscription."
                ]
            },
            "acknowledge": {
                "agent": [
                    "Certainly. I can help you review the subscription cancellation process.",
                    "I understand. Let us check the recurring payment details."
                ]
            },
            "verification_request": {
                "agent": [
                    "Please provide the relevant subscription details.",
                    "For security, I need to verify your account."
                ]
            },
            "verification_response": {
                "customer": [
                    "Sure, I can provide the information.",
                    "Yes, please proceed."
                ]
            },
            "resolve": {
                "agent": [
                    "The subscription cancellation request has been processed.",
                    "I have initiated the cancellation request."
                ]
            },
            "inform": {
                "agent": [
                    "You will receive confirmation once the cancellation is completed.",
                    "The recurring payment status will be updated after processing."
                ]
            },
            "confirm": {
                "customer": [
                    "Thank you for your help.",
                    "Okay, thank you."
                ]
            },
            "closing": {
                "agent": [
                    "You are welcome. Please contact us if you need further assistance.",
                    "Happy to help. Have a good day."
                ]
            }
        }
    },

    "FEE_CHARGED_QUERY": {
        "family": "Fees",

        "entities": {
            "fee_type": [
                "an annual fee",
                "a service fee",
                "a transaction fee",
                "an unexpected account charge"
            ]
        },

        "flow": [
            "problem_statement",
            "acknowledge",
            "verification_request",
            "verification_response",
            "status_check",
            "inform",
            "confirm",
            "closing"
        ],

        "templates": {
            "problem_statement": {
                "customer": [
                    "I noticed {fee_type} on my account.",
                    "Can you explain why I was charged {fee_type}?",
                    "I have a question about an unexpected fee."
                ]
            },
            "acknowledge": {
                "agent": [
                    "Certainly. I can help you review the fee.",
                    "I understand. Let me check the charge information."
                ]
            },
            "verification_request": {
                "agent": [
                    "For security, please complete account verification.",
                    "Before accessing the account details, I need to verify your identity."
                ]
            },
            "verification_response": {
                "customer": [
                    "Sure, please proceed.",
                    "Yes, I can verify my details."
                ]
            },
            "status_check": {
                "agent": [
                    "I have reviewed the account charge information.",
                    "Let me check the details associated with the fee."
                ]
            },
            "inform": {
                "agent": [
                    "The fee information has been reviewed and the applicable details are available.",
                    "I can provide the relevant explanation for the charge based on the account record."
                ]
            },
            "confirm": {
                "customer": [
                    "Thank you for explaining.",
                    "Okay, I understand."
                ]
            },
            "closing": {
                "agent": [
                    "You are welcome. Please contact us if you have further questions.",
                    "Happy to help. Have a good day."
                ]
            }
        }
    },

    "INTEREST_RATE_QUERY": {
        "family": "Account Information",

        "entities": {
            "rate_type": [
                "purchase interest rate",
                "cash advance interest rate",
                "applicable account interest rate"
            ]
        },

        "flow": [
            "problem_statement",
            "acknowledge",
            "verification_request",
            "verification_response",
            "inform",
            "confirm",
            "closing"
        ],

        "templates": {
            "problem_statement": {
                "customer": [
                    "What is my {rate_type}?",
                    "Can you tell me about the interest rate on my account?",
                    "I need information about my applicable interest rate."
                ]
            },
            "acknowledge": {
                "agent": [
                    "Certainly. I can help you review the interest rate information.",
                    "I can check the applicable rate details."
                ]
            },
            "verification_request": {
                "agent": [
                    "For account-specific information, please complete verification.",
                    "Before accessing your account details, I need to verify your identity."
                ]
            },
            "verification_response": {
                "customer": [
                    "Sure, please proceed.",
                    "Yes, I can verify my details."
                ]
            },
            "inform": {
                "agent": [
                    "The applicable {rate_type} information is available in your account details.",
                    "I have retrieved the relevant interest rate information."
                ]
            },
            "confirm": {
                "customer": [
                    "Thank you, that is helpful.",
                    "Okay, that answers my question."
                ]
            },
            "closing": {
                "agent": [
                    "You are welcome. Please contact us if you have more questions.",
                    "Happy to help. Have a good day."
                ]
            }
        }
    },

    "PAYMENT_DUE_DATE_QUERY": {
        "family": "Payments",

        "entities": {
            "due_date": [
                "the fifteenth of this month",
                "the twenty-second of this month",
                "the first working day of next month"
            ]
        },

        "flow": [
            "problem_statement",
            "acknowledge",
            "verification_request",
            "verification_response",
            "inform",
            "confirm",
            "closing"
        ],

        "templates": {
            "problem_statement": {
                "customer": [
                    "I would like to know when my next payment is due.",
                    "Can you tell me the due date for my next payment?",
                    "I need information about my upcoming payment due date."
                ]
            },
            "acknowledge": {
                "agent": [
                    "Certainly. I can help you check the payment due date.",
                    "I can look that up for you."
                ]
            },
            "verification_request": {
                "agent": [
                    "For verification, please confirm the requested account details.",
                    "Before I check the due date, I need to verify your account."
                ]
            },
            "verification_response": {
                "customer": [
                    "Sure, I can verify my details.",
                    "Yes, please proceed with verification."
                ]
            },
            "inform": {
                "agent": [
                    "Your next payment is due on {due_date}.",
                    "The payment due date is {due_date}."
                ]
            },
            "confirm": {
                "customer": [
                    "Thank you. That is what I needed.",
                    "Okay, thank you for confirming."
                ]
            },
            "closing": {
                "agent": [
                    "You are welcome. Is there anything else I can help you with?",
                    "Happy to help. Have a good day."
                ]
            }
        }
    },

    # ========================================================
    # ADDITIONAL SCENARIOS TO REACH 50
    # ========================================================

    "CARD_SECURITY_ALERT": {
        "family": "Card Security",

        "entities": {
            "alert_type": [
                "an unusual transaction alert",
                "a security notification",
                "a suspicious activity alert"
            ]
        },

        "flow": [
            "problem_statement",
            "acknowledge",
            "transaction_verification",
            "customer_response",
            "security_action",
            "inform",
            "confirm",
            "closing"
        ],

        "templates": {
            "problem_statement": {
                "customer": [
                    "I received {alert_type} and want to know what happened.",
                    "Can you help me understand {alert_type}?",
                    "I received a security alert related to my card."
                ]
            },
            "acknowledge": {
                "agent": [
                    "Certainly. I can help you review the security alert.",
                    "I understand your concern. Let us check the alert details."
                ]
            },
            "transaction_verification": {
                "agent": [
                    "Can you confirm whether you recognize the activity mentioned in the alert?",
                    "Please confirm whether the transaction was authorized by you."
                ]
            },
            "customer_response": {
                "customer": [
                    "I do not recognize the activity.",
                    "Yes, I recognize the transaction."
                ]
            },
            "security_action": {
                "agent": [
                    "I have taken the appropriate security action based on the information provided.",
                    "The account security status has been updated accordingly."
                ]
            },
            "inform": {
                "agent": [
                    "You will receive additional notifications if further action is required.",
                    "Please continue monitoring your account activity."
                ]
            },
            "confirm": {
                "customer": [
                    "Thank you for checking.",
                    "Okay, I understand."
                ]
            },
            "closing": {
                "agent": [
                    "You are welcome. Please contact us if you notice anything else unusual.",
                    "Happy to help. Have a good day."
                ]
            }
        }
    },

    "ACCOUNT_TRANSACTION_HISTORY": {
        "family": "Account Information",

        "entities": {
            "period": [
                "today",
                "the last seven days",
                "the previous month"
            ]
        },

        "flow": [
            "problem_statement",
            "acknowledge",
            "verification_request",
            "verification_response",
            "inform",
            "confirm",
            "closing"
        ],

        "templates": {
            "problem_statement": {
                "customer": [
                    "I would like to review my transaction history for {period}.",
                    "Can you help me check my recent account activity?",
                    "I need information about my transactions."
                ]
            },
            "acknowledge": {
                "agent": [
                    "Certainly. I can help you access the transaction history.",
                    "I can review the available transaction information."
                ]
            },
            "verification_request": {
                "agent": [
                    "For security, please complete account verification.",
                    "Before accessing the transaction history, I need to verify your identity."
                ]
            },
            "verification_response": {
                "customer": [
                    "Sure, please proceed.",
                    "Yes, I can verify my details."
                ]
            },
            "inform": {
                "agent": [
                    "The transaction history for {period} is available for review.",
                    "I have retrieved the requested account activity information."
                ]
            },
            "confirm": {
                "customer": [
                    "Thank you, that is helpful.",
                    "Great, thank you."
                ]
            },
            "closing": {
                "agent": [
                    "You are welcome. Is there anything else I can help you with?",
                    "Happy to help. Have a good day."
                ]
            }
        }
    }
}


# ============================================================
# OPTIONAL COMPATIBILITY CONSTANTS
# ============================================================

SCENARIO_NAMES = list(SCENARIOS.keys())


SCENARIO_FAMILIES = {
    scenario_name: scenario_data["family"]
    for scenario_name, scenario_data in SCENARIOS.items()
}


AGENT_PERSONAS = [
    "Priya",
    "Rahul",
    "Meera",
    "Daniel",
    "Ananya",
    "Victor",
    "Ishaan",
    "Grace",
    "Farhan",
    "Naomi"
]


CUSTOMER_PERSONAS = [
    "Arjun",
    "Neha",
    "Ravi",
    "Kavya",
    "Thomas",
    "Lena",
    "Sanjay",
    "Priyanka",
    "Omar",
    "Elena",
    "Vikram",
    "Maya",
    "Joseph",
    "Fatima",
    "Nikhil",
    "Clara"
]