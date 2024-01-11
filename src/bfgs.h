// Copyright © 2022 Alexander Abramenkov. All rights reserved.
// Licensed under the Apache License, Version 2.0.
// https://github.com/IOdissey/bfgs

#pragma once

#include <cmath>
#include <cstring>
#include <functional>
#include <limits>

#ifdef BFGS_AUTO
#include "dval.h"
#include "memory.h"
#endif

class BFGS
{
private:
	uint32_t _n = 0;                // Problem dimension.
	float* _ptr = nullptr;         // Memory for all value.
	float* _g = nullptr;           // The gradient value.
	float* _p = nullptr;           // Search direction.
	float* _d = nullptr;           // The gradient increment.
	float* _s = nullptr;           // Step.
	float* _h = nullptr;           // Approximation of the inverse Hessian or temporary array for lbfgs version.
	float* _ai = nullptr;          // Temporary array for lbfgs version.
	uint32_t _lbfgs_i = 0;          // History index for lbfgs version.

	float _min_f = std::numeric_limits<float>::infinity();
	float _stop_grad_eps = 1e-7;   // Stop criteria.
	float _stop_step_eps = 1e-9;   // Stop criteria.
	float _eps = 1e-8;             // Increment for calculating the derivative and another. TODO. Use different values.
	uint32_t _max_iter = 1000;      // Maximum number of algorithm iterations.
	float _c1 = 0.01;              // Constant for checking the Wolfe condition.
	float _c2 = 0.9;               // Constant for checking the Wolfe condition.
	uint32_t _line_iter_max = 100;  // Maximum number of linear search iterations.
	bool _central_diff = true;      // Central difference.
	bool _line_central_diff = true; // Central difference.
	const float _db = 2.0;         // Upper bound increment multiplier.
	float _step_tweak = 0.0;       // Initial step value tweak (if _step_tweak > 0.0).
	bool _reuse_hessian = false;    // Reuse hessian from last time.
	bool _select_hessian = false;   // Selection of the initial approximation of the inverse hessian.
	bool _memory_save = false;      // Store only the upper triangular matrix for the inverse hessian.
	bool _use_strong_wolfe = true;  // Use Strong Wolfe conditions.
	uint32_t _dval_size = 100;      // Specifies the amount of memory for automatic derivative with dynamic dimension.
	bool _line_force_num = false;   // Force use of numerical derivative in linear search for the case of an analytic derivative.
	uint32_t _lbfgs_m = 0;          // History size for lbfgs version. If 0 then the regular version is used.
	
	// Diagonal matrix.
	void _diag(float* const m, const float val = 1.0) const
	{
		if (_memory_save)
		{
			// | 0 4 7 9 |
			// | - 1 5 8 |
			// | - - 2 6 |
			// | - - - 3 |
			std::fill_n(m, _n, val);
			std::fill_n(m + _n, _n * (_n - 1) / 2, 0.0);
		}
		else
		{
			const uint32_t nn = _n * _n;
			const uint32_t step = _n + 1;
			std::fill_n(m, nn, 0.0);
			for (uint32_t i = 0; i < nn; i += step)
				m[i] = val;
		}
	}

	// L2 norm.
	float _norm(const float* const v) const
	{
		float r = 0.0;
		for (uint32_t i = 0; i < _n; ++i)
			r += v[i] * v[i];
		return std::sqrt(r);
	}

	// r = m * v.
	void _mull_m_v(const float* m, const float* const v, float* const r) const
	{
		if (_memory_save)
		{
			// | 0 4 7 9 |
			// | - 1 5 8 |
			// | - - 2 6 |
			// | - - - 3 |
			uint32_t k = 0;
			for (; k < _n; ++k)
				r[k] = m[k] * v[k];
			for (uint32_t i = 1; i < _n; ++i)
			{
				for (uint32_t j = i; j < _n; ++j, ++k)
				{
					r[j - i] += m[k] * v[j];
					r[j] += m[k] * v[j - i];
				}
			}
		}
		else
		{
			for (uint32_t i = 0; i < _n; ++i)
			{
				r[i] = 0.0;
				for (uint32_t j = 0; j < _n; ++j, ++m)
					r[i] += *m * v[j];
			}
		}
	}

	// r = v1 + a * v2.
	inline void _add_v_av(const float* const v1, const float* const v2, const float& a, float* const r) const
	{
		for (uint32_t i = 0; i < _n; ++i)
			r[i] = v1[i] + a * v2[i];
	}

	// v1 = v1 + a * v2
	inline void _add_v_av(float* const v1, const float* const v2, const float& a) const
	{
		for (uint32_t i = 0; i < _n; ++i)
			v1[i] += a * v2[i];
	}

	// r = v1^T * v2.
	inline float _mull_v_v(const float* const v1, const float* const v2) const
	{
		float r = 0.0;
		for (uint32_t i = 0; i < _n; ++i)
			r += v1[i] * v2[i];
		return r;
	}

	// r = v.
	inline void _copy_v(const float* const v, float* const r) const
	{
		std::memcpy(r, v, _n * sizeof(float));
	}

	// Pointer swap.
	inline void _swap_p(float*& v, float*& r)
	{
		float* tmp = v;
		v = r;
		r = tmp;
	}

	// r = v1 - v2.
	inline void _sub_v_v(const float* const v1, const float* const v2, float* const r) const
	{
		for (uint32_t i = 0; i < _n; ++i)
			r[i] = v1[i] - v2[i];
	}

	// v = v * a
	inline void _mull_v_a(float* const v, const float& a) const
	{
		for (uint32_t i = 0; i < _n; ++i)
			v[i] *= a;
	}

	//
	inline uint32_t _lbfgs_get_idx(const uint32_t& i) const
	{
		if (_lbfgs_i < i)
			return (_lbfgs_m + _lbfgs_i) - i;
		else
			return _lbfgs_i - i;
	}

	// Memory free.
	void _free_ptr()
	{
		if (!_ptr)
			return;
		delete[] _ptr;
		_ptr = nullptr;
		_n = 0;
	}

	// Memory init.
	void _init_ptr(uint32_t n)
	{
		if (n != _n)
		{
			_free_ptr();
			_n = n;
			if (_lbfgs_m > 0)
				// 2 * n + 2 * m * n + 2 * m = 2 * (n * m + n + m)
				_ptr = new float[2 * (_n * _lbfgs_m + _n + _lbfgs_m)];
			else if (_memory_save)
				// 4 * n + n * (n + 1) / 2 = n * (n + 9) / 2
				_ptr = new float[_n * (_n + 9) / 2];
			else
				// 4 * n + n * n = n * (n + 4)
				_ptr = new float[_n * (_n + 4)];
			_g = _ptr;
			_p = _g + _n;
			_d = _p + _n;
			if (_lbfgs_m > 0)
			{
				_s = _d + _lbfgs_m * _n;
				_h = _s + _lbfgs_m * _n;
				_ai = _h + _lbfgs_m;
				_lbfgs_i = 0;
			}
			else
			{
				_s = _d + _n;
				_h = _s + _n;
				_diag(_h);
			}
		}
		else if (_lbfgs_m > 0)
			_lbfgs_i = 0;
		else if (!_reuse_hessian)
			_diag(_h);
	}

	// Cubic approximation for finding the minimum.
	// There is always exists the minimum if ga < 0 and gb > 0.
	// f(x) = a0 + a1 * x + a2 * x^2 + a3 * x^3
	// f(a) = fa
	// f'(a) = da
	// f(b) = fb
	// f'(b) = db
	float _poly(
		const float a, const float b,
		const float fa, const float da,
		const float fb, const float db,
		const float limit_min = 0.0, const float limit_max = 1.0) const
	{
		// Search the minimum for the range [0, 1] and then scale to [a, b].
		// f(0) = fa
		// f'(0) = da
		// f(1) = fb
		// f'(1) = db
		const float& a0 = fa;
		const float& a1 = da;
		const float a2 = 3 * (fb - fa) - 2 * da - db;
		const float a3 = da + db - 2 * (fb - fa);
		// 
		float x_min = std::numeric_limits<float>::max();
		// f'(x) = a1 + 2 * a2 * x + 3 * a3 * x^2 = 0
		if (std::abs(a3) > _eps)
		{
			float d = a2 * a2 - 3 * a3 * a1;
			if (d >= 0.0)
			{
				d = std::sqrt(d);
				float x1 = (-a2 - d) / (3 * a3);
				float x2 = (-a2 + d) / (3 * a3);
				float y1 = a0 + (a1 + (a2 + a3 * x1) * x1) * x1;
				float y2 = a0 + (a1 + (a2 + a3 * x2) * x2) * x2;
				if (y1 < y2)
					x_min = x1;
				else
					x_min = x2;
			}
		}
		// f'(x) = a1 + 2 * a2 * x = 0
		else if (std::abs(a2) > _eps)
			x_min = -a1 / (2 * a2);
		if (x_min < 0.0 || x_min > 1.0)
			x_min = (limit_max + limit_max) / 2;
		else if (x_min < limit_min)
			x_min = limit_min;
		else if (x_min > limit_max)
			x_min = limit_max;
		// Apply scale to the result.
		return a + x_min * (b - a);
	}

	// Value and derivative of the function f(x + a * p).
	// f - function.
	// x - initial point.
	// p - search direction.
	// a - step value.
	// xa = x + a * p (result).
	// fa = f(x + a * p) (result).
	// ga = df(x + a * p) / da (result).
	void _f_info(
			const std::function<float (float*, uint32_t)>& f,
			const float* const x,
			const float* const p,
			const float a,
			float* const xa,
			float& fa,
			float& ga,
			float* const g)
	{
		if (_line_central_diff)
		{
			_add_v_av(x, p, a + _eps, xa);
			const float fp = f(xa, _n);
			_add_v_av(x, p, a - _eps, xa);
			const float fm = f(xa, _n);
			ga = (fp - fm) / (_eps + _eps);
			_add_v_av(x, p, a, xa);
			fa = f(xa, _n);
		}
		else
		{
			_add_v_av(x, p, a + _eps, xa);
			const float fp = f(xa, _n);
			_add_v_av(x, p, a, xa);
			fa = f(xa, _n);
			ga = (fp - fa) / _eps;
		}
	}

	void _f_info(
			const std::function<float (float*, float*, uint32_t)>& f,
			const float* const x,
			const float* const p,
			const float a,
			float* const xa,
			float& fa,
			float& ga,
			float* const g)
	{
		if (_line_force_num)
		{
			if (_line_central_diff)
			{
				_add_v_av(x, p, a + _eps, xa);
				const float fp = f(xa, nullptr, _n);
				_add_v_av(x, p, a - _eps, xa);
				const float fm = f(xa, nullptr, _n);
				ga = (fp - fm) / (_eps + _eps);
				_add_v_av(x, p, a, xa);
				fa = f(xa, nullptr, _n);
			}
			else
			{
				_add_v_av(x, p, a + _eps, xa);
				const float fp = f(xa, nullptr, _n);
				_add_v_av(x, p, a, xa);
				fa = f(xa, nullptr, _n);
				ga = (fp - fa) / _eps;
			}
		}
		else
		{
			_add_v_av(x, p, a, xa);
			fa = f(xa, g, _n);
			ga = _mull_v_v(g, p);
		}
	}

	inline float _f_info(const std::function<float (float*, uint32_t)>& f, float* const x, float* const g) const
	{
		float y = f(x, _n);
		_f_grad(f, y, x, _g);
		return y;
	}

	inline float _f_info(const std::function<float (float*, float*, uint32_t)>& f, float* const x, float* const g) const
	{
		return f(x, g, _n);
	}

	inline void _f_grad(const std::function<float (float*, uint32_t)>& f, const float& y, float* const x, float* const g) const
	{
		if (_central_diff)
		{
			for (uint32_t i = 0; i < _n; ++i)
			{
				const float x_i = x[i];
				x[i] = x_i + _eps;
				const float y_p = f(x, _n);
				x[i] = x_i - _eps;
				const float y_m = f(x, _n);
				x[i] = x_i;
				g[i] = (y_p - y_m) / (_eps + _eps);
			}
		}
		else
		{
			for (uint32_t i = 0; i < _n; ++i)
			{
				const float x_i = x[i];
				x[i] = x_i + _eps;
				const float y_p = f(x, _n);
				x[i] = x_i;
				g[i] = (y_p - y) / _eps;
			}
		}
	}

	inline void _f_grad(const std::function<float (float*, float*, uint32_t)>& f, const float& y, float* const x, float* const g) const
	{
		if (_line_force_num)
			f(x, g, _n);
	}

	// Line search.
	// f - function.
	// x - initial point.
	// p - search direction.
	// y = f(x).
	// d = grad(f(x))^T * p.
	// xi = x + ai * p - result point.
	// fi = f(xi) - function value in result point.
	// g = grad(f(xi)) (if the derivatives are analytical).
	template <typename fun>
	float _line_search(
			const fun& f,
			const float* const x,
			const float* const p,
			const float y,
			const float d,
			float* const xi,
			float& fi,
			float* const g)
	{
		// 0 < c1 < c2 < 1
		// Strong Wolfe conditions.
		// f(x + a * p) <= f(x) + c1 * a * grad(f(x))^T * p
		// |grad(f(x + a * p))^T * p| <= c2 * |grad(f(x))^T * p|
		// Normal Wolfe condition.
		// f(x + a * p) <= f(x) + c1 * a * grad(f(x))^T * p
		// grad(f(x + a * p))^T * p >= c2 * grad(f(x))^T * p
		// Preparing constants.
		// d = grad(f(x))^T * p
		// k1 = c1 * d = c1 * grad(f(x))^T * p
		// k2 = c2 * |d| = c2 * |grad(f(x))^T * p|
		const float k1 = _c1 * d;
		const float k2 = _use_strong_wolfe ? _c2 * std::abs(d) : _c2 * d;
		// Search the range [a, b], a < b.
		float a = 0.0;
		// fa = f(a)
		float fa = y;
		// ga = grad(f(a))^T * p
		float ga = d;
		// Initial value.
		float b = 1.0;
		// Step tweak.
		if (_step_tweak > 0.0 && y > _min_f)
		{
			b = _step_tweak * (_min_f - y) / d;
			if (b > 1.0)
				b = 1.0;
		}
		// fb = f(b)
		float fb;
		// gb = grad(f(b))^T * p
		float gb;
		//
		uint32_t line_iter = 0;
		for (; line_iter < _line_iter_max; ++line_iter)
		{
			_f_info(f, x, p, b, xi, fb, gb, g);
			// 
			if (fb > fa || fb > y + b * k1)
				break;
			// Wolfe conditions is satisfied.
			if ((_use_strong_wolfe && std::abs(gb) < k2) || (!_use_strong_wolfe && gb > k2))
			{
				fi = fb;
				return y - fi;
			}
			// Stop if ga < 0 and gb > 0.
			// There is the minimum in this range.
			if (gb > 0)
				break;
			// 
			a = b;
			fa = fb;
			ga = gb;
			// TODO. Extrapolation.
			// Simple way increase b.
			b *= _db;
		}
		//
		float gi;
		for (; line_iter < _line_iter_max; ++line_iter)
		{
			// TODO. Some thresholds to avoid extreme points.
			const float ai = _poly(a, b, fa, ga, fb, gb, 0.1, 0.9);
			_f_info(f, x, p, ai, xi, fi, gi, g);
			//
			if (fi > fa || fi > y + b * k1)
			{
				b = ai;
				fb = fi;
				gb = gi;
				continue;
			}
			// Wolfe conditions is satisfied.
			if ((_use_strong_wolfe && std::abs(gi) < k2) || (!_use_strong_wolfe && gi > k2))
				break;
			if (gi < 0)
			{
				a = ai;
				fa = fi;
				ga = gi;
			}
			else
			{
				b = ai;
				fb = fi;
				gb = gi;
			}
		}
		// TODO. Check fails (line_iter == _line_iter_max).
		return y - fi;
	}

public:
	BFGS()
	{
	}

	~BFGS()
	{
		_free_ptr();
	}

	// Stop criterion. Gradient Norm.
	// Defaul: 1e-7.
	void set_stop_grad_eps(float stop_grad_eps)
	{
		if (_stop_grad_eps > 0.0)
			_stop_grad_eps = stop_grad_eps;
	}

	// Stop criterion. The amount the feature changes between iterations.
	// Defaul: 1e-7.
	void set_stop_step_eps(float stop_step_eps)
	{
		if (stop_step_eps > 0.0)
			_stop_step_eps = stop_step_eps;
	}

	// Epsilon for calculating gradients.
	// It is also used for some other checks.
	// Defaul: 1e-8.
	void set_grad_eps(float grad_eps)
	{
		if (grad_eps > 0.0)
			_eps = grad_eps;
	}

	// Stop criterion. The maximum number of iterations of the algorithm.
	void set_max_iter(uint32_t max_iter)
	{
		if (max_iter > 0)
			_max_iter = max_iter;
	}

	// Wolfe criterion constants.
	// 0.0 < c1 < c2 < 1.0
	// Defaul: 0.01 and 0.9.
	void set_c1_c2(float c1, float c2)
	{
		if (c1 > 0.0 && c2 > c1 && c2 < 1.0)
		{
			_c1 = c1;
			_c2 = c2;
		}
	}

	// The maximum number of iterations of the line search.
	// Defaul: 100.
	void set_line_max_iter(uint32_t line_max_iter)
	{
		if (line_max_iter > 0)
			_line_iter_max = line_max_iter;
	}

	// Use central difference or forward difference for the gradient.
	// Defaul: true.
	void set_central_diff(bool central_diff)
	{
		_central_diff = central_diff;
	}

	// Use central difference or forward difference for the line serach.
	// Maybe we don't want too much precision here.
	// You can try it to reduce the number of function calls.
	// Defaul: true.
	void set_line_central_diff(bool line_central_diff)
	{
		_line_central_diff = line_central_diff;
	}

	// Estimated minimum of the function.
	// Used for stop criterion if fk - grad_eps < min_f
	// Also used for step_tweak option.
	// Defaul: inf (not use).
	void set_min_f(float min_f = std::numeric_limits<float>::infinity())
	{
		_min_f = min_f;
	}

	// Experimental setting of the first step of the linear search.
	// If set, then the first step of the linear search will be:
	// a = min(1, step_tweak * (f_min - f) / drad(f))
	// Defaul: 0.0 (not use if <= 0.0).
	void set_step_tweak(float step_tweak)
	{
		_step_tweak = step_tweak;
	}

	// Experimental setting.
	// When solving many typical problems, the value from the previous run
	// can be used as the initial approximation of the inverse Hessian.
	// Defaul: false.
	void set_reuse_hessian(bool reuse_hessian)
	{
		_reuse_hessian = reuse_hessian;
	}

	// If we want to save some memory for a large problem.
	// In the usual case,
	//    the memory size is proportional to n * (n + 4).
	// In the case of storing only the upper triangular inverse Hessian matrix,
	//    the memory size is proportional to n * (n + 9) / 2.
	// Defaul: false.
	void set_memory_save(bool memory_save)
	{
		if (memory_save != _memory_save)
		{
			_memory_save = memory_save;
			_free_ptr();
		}
	}

	// Selection of the initial approximation of the inverse hessian.
	// Default: false.
	void set_select_hessian(bool select_hessian)
	{
		_select_hessian = select_hessian;
	}

	// Use Strong Wolfe conditions.
	// Defaul: true.
	void set_strong_wolfe(bool use_strong_wolfe)
	{
		_use_strong_wolfe = use_strong_wolfe;
	}

	// Specifies the amount of memory for automatic derivative with dynamic dimension.
	// Defaul: 100.
	void set_dval_size(uint32_t dval_size)
	{
		_dval_size = dval_size;
	}

	// Force use of numerical derivative in linear search for the case of an analytic derivative.
	// Defaul: false.
	void set_line_force_num(bool line_force_num)
	{
		_line_force_num = line_force_num;
	}

	// History size for lbfgs version. If 0 then the regular version is used.
	// The memory size is proportional to 2 * (n * m + n + m).
	// Defaul: 0.
	void set_lbfgs_m(uint32_t lbfgs_m)
	{
		if (lbfgs_m != _lbfgs_m)
		{
			_lbfgs_m = lbfgs_m;
			_free_ptr();
		}
	}

	// Search for the function minimum.
	// f - minimized function
	//    numerical derivative: float f(float* x, uint32_t n)
	//    analytic derivative:  float f(float* x, float* g, uint32_t n)
	// x - initial point value and result.
	// n - dimension.
	template <typename fun>
	float find_min(const fun& f, float* const x, const uint32_t n)
	{
		//
		if (n < 1 || x == nullptr)
			return std::numeric_limits<float>::infinity();
		//
		_init_ptr(n);
		// Function value and derivatives.
		// y = f(x)
		// g = grad(f(x))
		float y = _f_info(f, x, _g);
		// LBFGS version.
		if (_lbfgs_m > 0)
		{
			// p = g
			_copy_v(_g, _p);
			// Limit the number of iterations.
			for (uint32_t iter = 0; iter < _max_iter; ++iter)
			{
				// p = -p
				for (uint32_t i = 0; i < _n; ++i)
					_p[i] = -_p[i];
				// d = g^T * p (the derivative at the point a = 0.0).
				const float d = _mull_v_v(_g, _p);
				if (d > -_eps)
					break;
				//
				float* si = _s + _lbfgs_i * _n;
				float* di = _d + _lbfgs_i * _n;
				// di = g
				_copy_v(_g, di);
				// si = x
				_copy_v(x, si);
				// Line search.
				// If the derivatives are analytical, then its value will be stored in _g.
				float dy = _line_search(f, si, _p, y, d, x, y, _g);
				// Stop check.
				if (std::abs(dy) < _stop_step_eps)
					break;
				// Stop check.
				if (std::isfinite(_min_f) && y - _eps < _min_f)
					break;
				// Calculation of the gradient.
				// g = grad(f(x))
				// If the derivatives are analytical and _line_force_num = false, then we do nothing.
				_f_grad(f, y, x, _g);
				// L2 norm of the gradient.
				// g_norm = |g|.
				// Stop check.
				if (_norm(_g) < _stop_grad_eps)
					break;
				// si = x - si
				_sub_v_v(x, si, si);
				// di = g - di
				_sub_v_v(_g, di, di);
				// sd = s^T * d
				const float sd = _mull_v_v(si, di);
				// TODO. break?
				if (std::abs(sd) < _eps)
					break;
				_h[_lbfgs_i] = 1.0 / sd;
				// Step update.
				// gamma = (s^T * d) / (d^T * d)
				const float gamma = sd / _mull_v_v(di, di);
				// p = g
				// _swap_p(_g, _p);
				_copy_v(_g, _p);
				//
				const uint32_t m = std::min(iter + 1, _lbfgs_m);
				for (uint32_t i = 0; i < m; ++i)
				{
					const uint32_t idx = _lbfgs_get_idx(i);
					si = _s + idx * _n;
					di = _d + idx * _n;
					// ai = hi * si^T * p
					_ai[i] = _h[idx] * _mull_v_v(si, _p);
					// p = p - ai * di
					_add_v_av(_p, di, -_ai[i]);
				}
				// p = gamma * p
				_mull_v_a(_p, gamma);
				//
				for (uint32_t i = m; i > 0; --i)
				{
					const uint32_t idx = _lbfgs_get_idx(i - 1);
					si = _s + idx * _n;
					di = _d + idx * _n;
					// b = hi * di^T * p
					const float b = _h[idx] * _mull_v_v(di, _p);
					// p = p + si * (ai - b)
					_add_v_av(_p, si, _ai[i - 1] - b);
				}
				++_lbfgs_i;
				if (_lbfgs_i >= _lbfgs_m)
					_lbfgs_i = 0;
			}
		}
		else
		{
			// Limit the number of iterations.
			uint32_t iter = 0;
			for (; iter < _max_iter; ++iter)
			{
				// p = h * g
				_mull_m_v(_h, _g, _p);
				// p = -p
				for (uint32_t i = 0; i < _n; ++i)
					_p[i] = -_p[i];
				// d = g^T * p (the derivative at the point a = 0.0).
				const float d = _mull_v_v(_g, _p);
				if (d > -_eps)
					break;
				// d = g
				// _swap_p(_g, _d);
				_copy_v(_g, _d);
				// s = x
				_copy_v(x, _s);
				// Line search.
				// If the derivatives are analytical, then its value will be stored in _g.
				float dy = _line_search(f, _s, _p, y, d, x, y, _g);
				// Stop check.
				if (std::abs(dy) < _stop_step_eps)
					break;
				// Stop check.
				if (std::isfinite(_min_f) && y - _eps < _min_f)
					break;
				// Calculation of the gradient.
				// g = grad(f(x))
				// If the derivatives are analytical and _line_force_num = false, then we do nothing.
				_f_grad(f, y, x, _g);
				// L2 norm of the gradient.
				// g_norm = |g|.
				// Stop check.
				if (_norm(_g) < _stop_grad_eps)
					break;
				// Update the inverse hessian.
				// h = h + (s^T * d + d^T * h * d) * (s * s^T) / (s^T * d)^2 - (h * d * s^T + s * d^T * h) / (s^T * d)
				// sd = 1 / (s^T * d) (scalar)
				// hd = h * d (vector)
				// h = h + sd * (1 + d^T * hd * sd) * (s * s^T) - (hd * s^T + s * hd^T) * sd
				// hd_sd = hd * sd (vector)
				// h = h + sd * (1 + d^T * hd_sd) * (s * s^T) - hd_sd * s^T - s * hd_sd^T
				// The gradient increment.
				// d = g - d.
				_sub_v_v(_g, _d, _d);
				// s = x - s
				_sub_v_v(x, _s, _s);
				// sd = s^T * d
				const float s_d = _mull_v_v(_s, _d);
				if (s_d < _eps)
					continue;
				// Selection of the initial approximation of the inverse hessian.
				// Equation (6.20) in the book "Numerical Optimization" by Nocedal and Wright (second edition).
				if (iter == 0 && _select_hessian && !_reuse_hessian)
				{
					const float d_d = _mull_v_v(_d, _d);
					float gamma = s_d / d_d;
					// TODO. Threshold?
					if (gamma < 1e-3)
						gamma = 1e-3;
					else if (gamma > 1e3)
						gamma = 1e3;
					_diag(_h, gamma);
				}
				const float sd = 1.0 / s_d;
				// p = hd = h * d = d^T * h (h - symmetric matrix)
				_mull_m_v(_h, _d, _p);
				// p = hd_sd = hd * sd
				_mull_v_a(_p, sd);
				// tmp = sd * (1 + d^T * hd_sd)
				const float tmp = sd * (1.0 + _mull_v_v(_d, _p));
				// h = h + tmp - hd_sd * s^T - s * hd_sd^T
				if (_memory_save)
				{
					// h - symmetric matrix
					// | 0 4 7 9 |
					// | - 1 5 8 |
					// | - - 2 6 |
					// | - - - 3 |
					uint32_t k = 0;
					for (; k < _n; ++k)
						_h[k] += _s[k] * (tmp * _s[k] - 2.0 * _p[k]);
					for (uint32_t i = 1; i < _n; ++i)
					{
						for (uint32_t j = i; j < _n; ++j, ++k)
							_h[k] += tmp * _s[j] * _s[j - i] - _p[j] * _s[j - i] - _s[j] * _p[j - i];
					}
				}
				else
				{
					for (uint32_t i = 0, m = 0; i < _n; ++i)
					{
						// h - symmetric matrix
						for (uint32_t j = 0; j < i; ++j, ++m)
							_h[m] = _h[j * _n + i];
						for (uint32_t j = i; j < _n; ++j, ++m)
							_h[m] += tmp * _s[i] * _s[j] - _p[i] * _s[j] - _s[i] * _p[j];
					}
				}
			}
			// If the algorithm fails.
			if (_reuse_hessian && iter == _max_iter)
				_diag(_h);
		}
		// y = f(х)
		return y;
	}

	float find_min_num(const std::function<float (float*, uint32_t)>& f, float* const x, const uint32_t n)
	{
		return find_min(f, x, n);
	}

	float find_min_grad(const std::function<float (float*, float*, uint32_t)>& f, float* const x, const uint32_t n)
	{
		return find_min(f, x, n);
	}

#ifdef BFGS_AUTO
	template <uint32_t N>
	float find_min_auto(const std::function<DVal<N> (DVal<N>*, uint32_t)>& f, float* const x, const uint32_t n = N)
	{
		if (n != N)
			return std::numeric_limits<float>::infinity();
		_line_force_num = false;
		DVal<N> dval[N];
		for (uint32_t i = 0; i < N; ++i)
			dval[i].set(x[i], i);
		DVal<N>* ptr = dval;
		auto g = [&f, ptr](const float* const x, float* const g, const uint32_t n) -> float
		{
			for (uint32_t i = 0; i < n; ++i)
				ptr[i].set(x[i]);
			const DVal<N> r = f(ptr, n);
			for (uint32_t i = 0; i < n; ++i)
				g[i] = r.d[i + 1];
			return r.d[0];
		};
		return find_min(g, x, n);
	}

	float find_min_auto(const std::function<DVal<0> (DVal<0>*, uint32_t)>& f, float* const x, const uint32_t n)
	{
		if (n == 0)
			return std::numeric_limits<float>::infinity();
		_line_force_num = false;
		Memory mem(n, _dval_size);
		auto g = [&mem, &f](const float* const x, float* const g, uint32_t n) -> float
		{
			DVal<0>* dval = new DVal<0>[n];
			for (uint32_t i = 0; i < n; ++i)
				dval[i].set(x[i], i, &mem);
			DVal<0> r = f(dval, n);
			delete[] dval;
			for (uint32_t i = 0; i < n; ++i)
				g[i] = r.d[i + 1];
			return r.d[0];
		};
		return find_min(g, x, n);
	}
#endif
};