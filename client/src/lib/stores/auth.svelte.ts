import { authAPI } from '../api/auth-api';
import type { User, AuthState, LoginRequest, RegisterRequest } from '../types/auth';

function createAuthStore() {
  let state = $state<AuthState>({
    user: null,
    isAuthenticated: false,
    isLoading: true,
  });

  return {
    // Getters
    get user() { return state.user; },
    get isAuthenticated() { return state.isAuthenticated; },
    get isLoading() { return state.isLoading; },

    // Actions
    async checkAuth() {
      try {
        state.isLoading = true;
        const response = await authAPI.checkAuthStatus();
        
        if (response.authenticated && response.user) {
          state.user = response.user;
          state.isAuthenticated = true;
        } else {
          state.user = null;
          state.isAuthenticated = false;
        }
      } catch (error) {
        console.error('Auth check failed:', error);
        state.user = null;
        state.isAuthenticated = false;
      } finally {
        state.isLoading = false;
      }
    },

    async login(credentials: LoginRequest): Promise<User> {
      try {
        const user = await authAPI.login(credentials);
        state.user = user;
        state.isAuthenticated = true;
        return user;
      } catch (error) {
        state.user = null;
        state.isAuthenticated = false;
        throw error;
      }
    },

    async register(userData: RegisterRequest): Promise<User> {
      try {
        const user = await authAPI.register(userData);
        state.user = user;
        state.isAuthenticated = true;
        return user;
      } catch (error) {
        state.user = null;
        state.isAuthenticated = false;
        throw error;
      }
    },

    async logout() {
      try {
        await authAPI.logout();
      } catch (error) {
        console.error('Logout error:', error);
      } finally {
        state.user = null;
        state.isAuthenticated = false;
      }
    },

    async updateProfile(userData: Partial<RegisterRequest>): Promise<User> {
      try {
        const user = await authAPI.updateProfile(userData);
        state.user = user;
        return user;
      } catch (error) {
        throw error;
      }
    },

    async refreshToken() {
      try {
        await authAPI.refreshToken();
      } catch (error) {
        console.error('Token refresh failed:', error);
        // On token refresh failure, logout user
        state.user = null;
        state.isAuthenticated = false;
        throw error;
      }
    }
  };
}

export const authStore = createAuthStore();