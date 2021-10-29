"""

    Kurs:       Einführung in das wissenschaftliche Rechnen
    Aufgabe:    Aufgabe 2.1
    Programm:   Input:  Ganze Zahl zwischen 1 und 12
                Output: Anzahl der Tage für den entsprechenden Monat
    Authoren:   Aron Ventura, Annika Thiele

"""

def tage_im_monat(zahl):
    """

    Funktion bestimmt Anzahl Tage zu entsprechendem Monat

    """
    return Monat.get(zahl, "Z")

Monat = {
    1: "31",
    2: "28",
    3: "31",
    4: "30",
    5: "31",
    6: "30",
    7: "31",
    8: "31",
    9: "30",
    10: "31",
    11: "30",
    12: "31",
}

def exception(j):
    """

    Funktion testet ob Zahl zwischen 1 und 12 übergeben wurde

    """
    i=0
    if j.isdigit():
        zahl=int(j)
        if 0<zahl<13:
            i=1
    return i

def main():
    """

     Liest Input ein, prüft auf Eingabefehler und übergibt die Zahl
     an die Funktion, die die Anzahl Tage im entsprechenden Monat
     bestimmt und gibt diese anschließend aus

    """
    zahlen=input("Bitte geben sie eine ganze Zahl zwischen 1 und 12 ein.")
    while not exception(zahlen)==1:
        print("Versuchen Sie es noch einmal.")
        zahlen=input("Bitte geben sie eine ganze Zahl zwischen 1 und 12 ein.")

    zahl=int(zahlen)

    print("Der " + str(zahl) + ". Monat hat" , tage_im_monat(zahl) ,
    "Tage.")




if __name__ == "__main__":
    main()
