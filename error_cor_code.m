EbNodb = 8;
R = 1;
EbNo = 10^(EbNodb/10);
sigma = sqrt(1/(2*R*EbNo));

BER_th = float('double');
BER_th = 0.5*erfc(sqrt(EbNo));

N = 10000;
N_blocks = 100;
Nerr = 0;
for i = 1:N_blocks
    msg = randi([0 1],1,N);
    s = 1-2*msg;
    r = s + sigma*randn(1,N);
    msg_cap = (r<0);

    Nerr = Nerr + sum(msg~=msg_cap); 
end
BER_sim = Nerr/(N*N_blocks);    
    
disp([ EbNodb BER_th BER_sim ]);