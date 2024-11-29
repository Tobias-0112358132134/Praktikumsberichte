import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Parameter für die Normalverteilung der Reaktionszeit t
mu = 0.170796268  # Mittelwert (Sekunden)
sigma = 0.015147979  # Standardabweichung

mu_h_act = 14.42028986 / 100  # actual mean value of h
sdtdev_h_act = 2.545933051 / 100  # actual standard deviation of h

# Gravitationskonstante
g = 9.81  # m/s²

# Bereich der Reaktionszeiten
t_values = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 1000)

# Normalverteilung der Reaktionszeit
t_pdf = norm.pdf(t_values, mu, sigma)

# Berechnung der entsprechenden Längen h
h_values = (g * t_values**2) / 2

# Transformation der Wahrscheinlichkeiten auf h
# dP(t) = pdf(t) dt -> pdf(h) = pdf(t) * |dt/dh|
dh_dt = g * t_values  # Ableitung von h nach t
h_pdf = t_pdf / np.abs(dh_dt)  # Transformation der Dichtefunktion

# Bereich der Längen für Darstellung
h_min, h_max = h_values.min(), h_values.max()
h_range = np.linspace(h_min, h_max, 1000)

# Interpolation der Dichte von h für eine glatte Kurve
h_pdf_interpolated = np.interp(h_range, h_values, h_pdf)

# Gaussian approximation for h
h_gaussian_pdf = norm.pdf(h_range, mu_h_act, sdtdev_h_act)

# Plot der Verteilungen
plt.figure(figsize=(8, 6))

# Reaktionszeit-Verteilung
# plt.plot(t_values, t_pdf, label="Normalverteilung von $t$", color="blue")

# Längen-Verteilung
plt.plot(h_range, h_pdf_interpolated, label="tatsächliche Verteilung von $h$", color="red")

# Gaussian approximation of h
plt.plot(h_range, h_gaussian_pdf, label="Normalvert.-Approximation von $h$", color="red", linestyle='dashed')

# # line at the actual mean value of h
# plt.axvline(x=mu_h_act, color='black', linestyle='dotted', label='tatsächlicher Erwartungswert von $h$')

# # line at the corresponding h value for the mean value of t
# plt.axvline(x=(g * mu**2) / 2, color='blue', linestyle='dotted', label='Erwartungswert von $h$')

plt.xlabel('$h$')
plt.ylabel('Wahrscheinlichkeitsdichte')
plt.legend()
plt.title('Vergleich der Verteilungen von $h$')
plt.grid(True)

plt.gca().set_xticklabels([])
plt.gca().set_yticklabels([])

plt.savefig('py-plot/plot.png', dpi=400)

# plt.show()
