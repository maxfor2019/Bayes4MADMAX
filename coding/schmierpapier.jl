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