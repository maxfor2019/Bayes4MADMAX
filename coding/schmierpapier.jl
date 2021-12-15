# Gaussian noise
x = rand(Normal(0,1), 1000)


function makeMatrix(x)
    xM = x
    for i in range(1,length(x)-1, step=1)
        x = append!(x[2:end], x[1])
        xM = hcat(xM, x)
    end   
    return xM 
end 

xM = reshape(x,(100,100))
cov(xM)

heatmap(cov(xM))

# wiener process
wp = [0.0]

for elem in x
    append!(wp, wp[end]+elem)
end

plot(wp)

wpM = makeMatrix(wp)
heatmap(cov(wpM))

# integrated wiener process
iwp = [0.0]

for elem in wp
    append!(iwp, iwp[end]+elem)
end

iwp = iwp[3:end]

plot(iwp)

iwpM = makeMatrix(iwp)
ciwpM = cov(iwpM)
heatmap(ciwpM)

iciwpM = inv(ciwpM)

using FFTW

ht = FFTW.plan_r2r(zeros(length(iwp)), FFTW.DHT);

hiwp = ht * iwp

arg = iciwpM * hiwp

harg = ht * arg

plot(harg[2:end])

sig = 218 * 1e3 / 3e8
vlab = 242.1 * 1e3 / 3e8
vearth = 29.8  * 1e3 / 3e8
σlab = 2 * 1e3 / 3e8

r2 = rand(MaxwellBoltzmann2(sig, vlab-vearth),10000)
r3 = rand(MaxwellBoltzmann2(sig, vlab+vearth),10000)
r1 = rand(MaxwellBoltzmann(sig),10000)

histogram(r1, alpha=0.5, label="normal MaxwellBoltzmann")
histogram!(r2, alpha=0.5, label="boosted MaxwellBoltzmann - earth")
histogram!(r3, alpha=0.5, label="boosted MaxwellBoltzmann + earth")

x = 0.0:0.00001:0.005
plot(x, pdf(MaxwellBoltzmann(sig),x), label="normal MaxwellBoltzmann")
plot!(x, pdf(MaxwellBoltzmann2(sig,vlab-vearth),x), label="boosted MaxwellBoltzmann -- earth")
plot!(x, pdf(MaxwellBoltzmann2(sig,vlab+vearth),x), label="boosted MaxwellBoltzmann + earth")




