import numpy as np 
import matplotlib.pyplot as plt 
import ot 
from scipy.stats import norm 
from scipy.linalg import sqrtm
from scipy.special import gamma


d = 100
n_samples = [1,10,100,200,500,1000,1500,2000,2500,3000,4000,5000,6000,7000,8000,10000]
mu1 = np.zeros(d)
mu2 = np.ones(d)
sigma1 = np.eye(d)
sigma2 = 4*np.eye(d)
n_projection = 1000
n_trials = 50

def surface_sphere(d):
    return 2 * np.pi ** (d / 2) / gamma(d / 2)

def _ensure_arrays(x,y): #Utility function
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if x.ndim == 1:
        x = x[:, None]
    if y.ndim == 1:
        y = y[:, None]
    return x, y


def wasserstein_p(x, y,p = 2, pth_power = False) : 
    if p < 1:
        raise ValueError(f"p must be >= 1, got {p}")
    x, y = _ensure_arrays(x,y)
    C = ot.dist(x,y,metric="euclidean")**p
    n,m = len(x), len(y)
    a, b = np.ones(n)/n, np.ones(m)/m
    wp_p = float(ot.emd2(a,b,C))
    return wp_p if pth_power else wp_p ** (1.0 / p)

def wasserstein_gaussian(mu1, mu2, sigma1,sigma2) : 
    mu1, mu2 = np.atleast_1d(mu1), np.atleast_1d(mu2)
    sigma1, sigma2 = np.atleast_2d(sigma1), np.atleast_2d(sigma2)
    diff_mu = mu1 - mu2
    mean_term = np.dot(diff_mu, diff_mu)
    sqrt_sigma1 = sqrtm(sigma1)
    product = sqrt_sigma1 @ sigma2 @ sqrt_sigma1
    sqrt_product = sqrtm(product)
    cov_term = np.real(np.trace(sigma1 + sigma2 - 2 * sqrt_product))

    return np.sqrt(mean_term + cov_term)


def sliced_wasserstein(x,y,n_projections =50, p = 2, pth_power = False) : 
    if p < 1:
        raise ValueError(f"p must be >= 1, got {p}")
    x, y = _ensure_arrays(x, y)
    n, d = x.shape
    m = y.shape[0]

    # 1) Sample directions sur la sphère — vectorisé
    directions = np.random.normal(0, 1, size=(n_projections, d))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)

    # 2) Projection de TOUS les points sur TOUTES les directions en un seul matmul
    proj_x = x @ directions.T   # shape (n, L)
    proj_y = y @ directions.T   # shape (m, L)

    # 3) Tri 1D vectorisé le long de l'axe des échantillons
    proj_x = np.sort(proj_x, axis=0)
    proj_y = np.sort(proj_y, axis=0)

    if n == m:
        w_p_per_proj = np.mean(np.abs(proj_x - proj_y) ** p, axis=0)
    else:
        u_levels = np.arange(1, n + 1) / n
        v_levels = np.arange(1, m + 1) / m
        levels = np.unique(np.concatenate([u_levels, v_levels]))
        widths = np.diff(np.concatenate(([0.0], levels)))
        u_idx = np.searchsorted(u_levels, levels, side='left')
        v_idx = np.searchsorted(v_levels, levels, side='left')
        diffs_p = np.abs(proj_x[u_idx] - proj_y[v_idx]) ** p   
        w_p_per_proj = (widths[:, None] * diffs_p).sum(axis=0)

    sw_p = w_p_per_proj.mean()
    return sw_p if pth_power else sw_p ** (1.0 / p)


def error(u,v,val,n_proj) : 
    return np.abs(wasserstein_p(u,v)-val), np.abs(sliced_wasserstein(u,v,n_projections=n_proj)-val/np.sqrt(d))


def invsqrt(x,d) : 
    return np.array([el**(-1/d) for el in x])

n_max = int(np.max(n_samples))
u= np.array([np.random.multivariate_normal(mu1,sigma1) for i in range(n_max)])
v = np.array([np.random.multivariate_normal(mu2,sigma2) for i in range(n_max)])
true_val = wasserstein_gaussian(mu1,mu2,sigma1,sigma2)
error_wass = np.zeros(len(n_samples))
error_slice = np.zeros(len(n_samples))
for _ in range(n_trials) : 
    for i,n in enumerate(n_samples) : 
        index_sample_u = np.random.choice(np.shape(u)[0], size = n, replace = False)
        index_sample_v = np.random.choice(np.shape(v)[0],size = n, replace = False)
        u_sample = u[index_sample_u]
        v_sample = v[index_sample_v]
        errw,errsl = error(u_sample,v_sample,true_val,n_projection)
        error_wass[i] += errw
        error_slice[i] += errsl
error_wass /=n_trials
error_slice /=n_trials



plt.loglog(n_samples, error_wass,'o-',label = "Wasserstein distance error")
#plt.loglog(n_samples, np.array(n_samples)**(-1/d),'o-',label = "Wasserstein theoretical error")
plt.loglog(n_samples, error_slice,'o-',label = "Sliced Wasserstein distance error")
plt.loglog(n_samples,np.array(n_samples)**(-1/2),'o-',label = "S.Wasserstein theoretical error")
plt.legend()
plt.tight_layout()
plt.show()