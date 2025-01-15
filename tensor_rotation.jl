function rotate!(n, R, C)
    ## Tensor
    a = C[1,1]
    b = C[2,2]
    c = C[3,3]
    d = C[4,4]
    e = C[5,5]
    f = C[6,6]
    g = C[1,2]
    h = C[1,3]
    l = C[2,3]
    ## Rotation matrix
    x1 = R[1,1]
    x2 = R[1,2]
    x3 = R[1,3]
    y1 = R[2,1]
    y2 = R[2,2]
    y3 = R[2,3]
    z1 = R[3,1]
    z2 = R[3,2]
    z3 = R[3,3]
    ## Rotated components
    n[1,1] = 4 * ((e * x3 ^ 2 + f * x2 ^ 2) * x1*x1 + d * x2 ^ 2 * x3 ^ 2) + (h * x1*x1 + l * x2 ^ 2 + c * x3 ^ 2) * x3 ^ 2 + (g * x2 ^ 2 + h * x3 ^ 2 + a * x1*x1) * x1*x1 + (g * x1*x1 + l * x3 ^ 2 + b * x2 ^ 2) * x2 ^ 2
    n[1,2] = n[1,2] = 4 * ((e * x3 * y3 + f * x2 * y2) * x1 * y1 + d * x2 * x3 * y2 * y3) + (h * x1*x1 + l * x2 ^ 2 + c * x3 ^ 2) * y3 ^ 2 + (g * x2 ^ 2 + h * x3 ^ 2 + a * x1*x1) * y1*y1 + (g * x1*x1 + l * x3 ^ 2 + b * x2 ^ 2) * y2*y2
    n[2,2] = 4 * ((e * y3 ^ 2 + f * y2*y2) * y1*y1 + d * y2*y2 * y3 ^ 2) + (h * y1*y1 + l * y2*y2 + c * y3 ^ 2) * y3 ^ 2 + (g * y2*y2 + h * y3 ^ 2 + a * y1*y1) * y1*y1 + (g * y1*y1 + l * y3 ^ 2 + b * y2*y2) * y2*y2
    n[1,3] = n[1,3] = 4 * ((e * x3 * z3 + f * x2 * z2) * x1 * z1 + d * x2 * x3 * z2 * z3) + (h * x1*x1 + l * x2 ^ 2 + c * x3 ^ 2) * z3 ^ 2 + (g * x2 ^ 2 + h * x3 ^ 2 + a * x1*x1) * z1 ^ 2 + (g * x1*x1 + l * x3 ^ 2 + b * x2 ^ 2) * z2 ^ 2
    n[2,3] = n[3,2] = 4 * ((e * y3 * z3 + f * y2 * z2) * y1 * z1 + d * y2 * y3 * z2 * z3) + (h * y1*y1 + l * y2*y2 + c * y3 ^ 2) * z3 ^ 2 + (g * y2*y2 + h * y3 ^ 2 + a * y1*y1) * z1 ^ 2 + (g * y1*y1 + l * y3 ^ 2 + b * y2*y2) * z2 ^ 2
    n[3,3] = 4 * ((e * z3 ^ 2 + f * z2 ^ 2) * z1 ^ 2 + d * z2 ^ 2 * z3 ^ 2) + (h * z1 ^ 2 + l * z2 ^ 2 + c * z3 ^ 2) * z3 ^ 2 + (g * z2 ^ 2 + h * z3 ^ 2 + a * z1 ^ 2) * z1 ^ 2 + (g * z1 ^ 2 + l * z3 ^ 2 + b * z2 ^ 2) * z2 ^ 2
    n[1,4] = n[1,4] = 2 * (((x2 * z3 + x3 * z2) * e * x1 + (y2 * z3 + y3 * z2) * d * x2) * x3 + (x2 * y3 + x3 * y2) * f * x1 * x2) + (h * x1*x1 + l * x2 ^ 2 + c * x3 ^ 2) * y3 * z3 + (g * x2 ^ 2 + h * x3 ^ 2 + a * x1*x1) * y1 * z1 + (g * x1*x1 + l * x3 ^ 2 + b * x2 ^ 2) * y2 * z2
    n[2,4] = n[4,2] = 2 * (((x2 * z3 + x3 * z2) * e * y1 + (y2 * z3 + y3 * z2) * d * y2) * y3 + (x2 * y3 + x3 * y2) * f * y1 * y2) + (h * y1*y1 + l * y2*y2 + c * y3 ^ 2) * y3 * z3 + (g * y2*y2 + h * y3 ^ 2 + a * y1*y1) * y1 * z1 + (g * y1*y1 + l * y3 ^ 2 + b * y2*y2) * y2 * z2
    n[3,4] = n[4,3] = 2 * (((x2 * z3 + x3 * z2) * e * z1 + (y2 * z3 + y3 * z2) * d * z2) * z3 + (x2 * y3 + x3 * y2) * f * z1 * z2) + (h * z1 ^ 2 + l * z2 ^ 2 + c * z3 ^ 2) * y3 * z3 + (g * z2 ^ 2 + h * z3 ^ 2 + a * z1 ^ 2) * y1 * z1 + (g * z1 ^ 2 + l * z3 ^ 2 + b * z2 ^ 2) * y2 * z2
    n[4,4] = (x2 * z3 + x3 * z2) ^ 2 * e + (y2 * z3 + y3 * z2) ^ 2 * d + (x2 * y3 + x3 * y2) ^ 2 * f + (h * y1 * z1 + l * y2 * z2 + c * y3 * z3) * y3 * z3 + (g * y2 * z2 + h * y3 * z3 + a * y1 * z1) * y1 * z1 + (g * y1 * z1 + l * y3 * z3 + b * y2 * z2) * y2 * z2
    n[1,5] = n[5,1] = 2 * (((x1 * z3 + x3 * z1) * e * x1 + (y1 * z3 + y3 * z1) * d * x2) * x3 + (x1 * y3 + x3 * y1) * f * x1 * x2) + (h * x1*x1 + l * x2 ^ 2 + c * x3 ^ 2) * x3 * z3 + (g * x2 ^ 2 + h * x3 ^ 2 + a * x1*x1) * x1 * z1 + (g * x1*x1 + l * x3 ^ 2 + b * x2 ^ 2) * x2 * z2
    n[2,5] = n[5,2] = 2 * (((x1 * z3 + x3 * z1) * e * y1 + (y1 * z3 + y3 * z1) * d * y2) * y3 + (x1 * y3 + x3 * y1) * f * y1 * y2) + (h * y1*y1 + l * y2*y2 + c * y3 ^ 2) * x3 * z3 + (g * y2*y2 + h * y3 ^ 2 + a * y1*y1) * x1 * z1 + (g * y1*y1 + l * y3 ^ 2 + b * y2*y2) * x2 * z2
    n[3,5] = n[5,3] = 2 * (((x1 * z3 + x3 * z1) * e * z1 + (y1 * z3 + y3 * z1) * d * z2) * z3 + (x1 * y3 + x3 * y1) * f * z1 * z2) + (h * z1 ^ 2 + l * z2 ^ 2 + c * z3 ^ 2) * x3 * z3 + (g * z2 ^ 2 + h * z3 ^ 2 + a * z1 ^ 2) * x1 * z1 + (g * z1 ^ 2 + l * z3 ^ 2 + b * z2 ^ 2) * x2 * z2
    n[4,5] = n[5,4] = (x1 * z3 + x3 * z1) * (x2 * z3 + x3 * z2) * e + (y1 * z3 + y3 * z1) * (y2 * z3 + y3 * z2) * d + (x1 * y3 + x3 * y1) * (x2 * y3 + x3 * y2) * f + (h * y1 * z1 + l * y2 * z2 + c * y3 * z3) * x3 * z3 + (g * y2 * z2 + h * y3 * z3 + a * y1 * z1) * x1 * z1 + (g * y1 * z1 + l * y3 * z3 + b * y2 * z2) * x2 * z2
    n[5,5] = (x1 * z3 + x3 * z1) ^ 2 * e + (y1 * z3 + y3 * z1) ^ 2 * d + (x1 * y3 + x3 * y1) ^ 2 * f + (h * x1 * z1 + l * x2 * z2 + c * x3 * z3) * x3 * z3 + (g * x2 * z2 + h * x3 * z3 + a * x1 * z1) * x1 * z1 + (g * x1 * z1 + l * x3 * z3 + b * x2 * z2) * x2 * z2
    n[1,6] = n[6,1] = 2 * (((x1 * z2 + x2 * z1) * e * x1 + (y1 * z2 + y2 * z1) * d * x2) * x3 + (x1 * y2 + x2 * y1) * f * x1 * x2) + (h * x1*x1 + l * x2 ^ 2 + c * x3 ^ 2) * x3 * y3 + (g * x2 ^ 2 + h * x3 ^ 2 + a * x1*x1) * x1 * y1 + (g * x1*x1 + l * x3 ^ 2 + b * x2 ^ 2) * x2 * y2
    n[2,6] = n[6,2] = 2 * (((x1 * z2 + x2 * z1) * e * y1 + (y1 * z2 + y2 * z1) * d * y2) * y3 + (x1 * y2 + x2 * y1) * f * y1 * y2) + (h * y1*y1 + l * y2*y2 + c * y3 ^ 2) * x3 * y3 + (g * y2*y2 + h * y3 ^ 2 + a * y1*y1) * x1 * y1 + (g * y1*y1 + l * y3 ^ 2 + b * y2*y2) * x2 * y2
    n[3,6] = n[6,3] = 2 * (((x1 * z2 + x2 * z1) * e * z1 + (y1 * z2 + y2 * z1) * d * z2) * z3 + (x1 * y2 + x2 * y1) * f * z1 * z2) + (h * z1 ^ 2 + l * z2 ^ 2 + c * z3 ^ 2) * x3 * y3 + (g * z2 ^ 2 + h * z3 ^ 2 + a * z1 ^ 2) * x1 * y1 + (g * z1 ^ 2 + l * z3 ^ 2 + b * z2 ^ 2) * x2 * y2
    n[4,6] = n[6,4] = (x1 * z2 + x2 * z1) * (x2 * z3 + x3 * z2) * e + (y1 * z2 + y2 * z1) * (y2 * z3 + y3 * z2) * d + (x1 * y2 + x2 * y1) * (x2 * y3 + x3 * y2) * f + (h * y1 * z1 + l * y2 * z2 + c * y3 * z3) * x3 * y3 + (g * y2 * z2 + h * y3 * z3 + a * y1 * z1) * x1 * y1 + (g * y1 * z1 + l * y3 * z3 + b * y2 * z2) * x2 * y2
    n[5,6] = n[6,5] = (x1 * z2 + x2 * z1) * (x1 * z3 + x3 * z1) * e + (y1 * z2 + y2 * z1) * (y1 * z3 + y3 * z1) * d + (x1 * y2 + x2 * y1) * (x1 * y3 + x3 * y1) * f + (h * x1 * z1 + l * x2 * z2 + c * x3 * z3) * x3 * y3 + (g * x2 * z2 + h * x3 * z3 + a * x1 * z1) * x1 * y1 + (g * x1 * z1 + l * x3 * z3 + b * x2 * z2) * x2 * y2
    n[6,6] = (x1 * z2 + x2 * z1) ^ 2 * e + (y1 * z2 + y2 * z1) ^ 2 * d + (x1 * y2 + x2 * y1) ^ 2 * f + (h * x1 * y1 + l * x2 * y2 + c * x3 * y3) * x3 * y3 + (g * x2 * y2 + h * x3 * y3 + a * x1 * y1) * x1 * y1 + (g * x1 * y1 + l * x3 * y3 + b * x2 * y2) * x2 * y2
    nothing
end

θ    = 30  
R    = [ # rotation along y-axis only
    cosd(θ)  0 sind(θ) 
    0        1 0      
    -sind(θ) 0 cosd(θ)
]
C    = zeros(6,6)
C[1,1] = C[2, 2] = 1
C[1,2] = C[2, 1] = 2
C[1,3] = C[3, 1] = C[2,3] = C[3, 2] = 3
C[3,3] = 4
C[4,4] = C[5, 5] = 5
C[6,6] = (C[1,1] - C[2,2]) / 2

Crot = zeros(6,6)

rotate!(Crot, R, C)
Crot