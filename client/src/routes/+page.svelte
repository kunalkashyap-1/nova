<script lang="ts">
	import { Camera, User } from 'lucide-svelte';
	import { authStore } from '$lib/stores/auth.svelte';
	import { goto } from '$app/navigation';
	import { onMount } from 'svelte';
	import type { LoginFormData, RegisterFormData } from '$lib/types/auth';

	let isLogin: boolean = $state(true);
	let isLoading: boolean = $state(false);
	let error: string = $state('');

	let loginFormData: LoginFormData = $state({
		username: '',
		password: ''
	});

	let registerFormData: RegisterFormData = $state({
		fullName: '',
		email: '',
		username: '',
		password: '',
		bio: '',
		preferredLanguage: '',
		timezone: ''
	});

	let profilePictureUrl: string = $state('');
	let fileInput: HTMLInputElement | null = $state(null);

	onMount(() => {
		// Check if user is already authenticated
		if (authStore.isAuthenticated) {
			goto('/chat');
		}
	});

	async function handleLogin(event: Event): Promise<void> {
		event.preventDefault();
		error = '';
		isLoading = true;

		try {
			await authStore.login({
				username: loginFormData.username,
				password: loginFormData.password
			});

			// Redirect to chat page on successful login
			goto('/chat');
		} catch (err) {
			error = err instanceof Error ? err.message : 'Login failed';
		} finally {
			isLoading = false;
		}
	}

	async function handleRegister(event: Event): Promise<void> {
		event.preventDefault();
		error = '';
		isLoading = true;

		try {
			await authStore.register({
				full_name: registerFormData.fullName,
				email: registerFormData.email,
				username: registerFormData.username,
				password: registerFormData.password,
				bio: registerFormData.bio,
				preferred_language: registerFormData.preferredLanguage,
				timezone: registerFormData.timezone,
				profile_picture: registerFormData.profilePicture
			});

			// Redirect to chat page on successful registration
			goto('/chat');
		} catch (err) {
			error = err instanceof Error ? err.message : 'Registration failed';
		} finally {
			isLoading = false;
		}
	}

	function handleFileChange(event: Event): void {
		const input = event.target as HTMLInputElement;
		if (input.files && input.files.length > 0) {
			registerFormData.profilePicture = input.files[0];
			profilePictureUrl = URL.createObjectURL(registerFormData.profilePicture);
		}
	}

	function triggerFileInput(): void {
		fileInput?.click();
	}

	function switchToLogin(): void {
		isLogin = true;
		error = '';
		resetForms();
	}

	function switchToRegister(): void {
		isLogin = false;
		error = '';
		resetForms();
	}

	function resetForms(): void {
		loginFormData = {
			username: '',
			password: ''
		};

		registerFormData = {
			fullName: '',
			email: '',
			username: '',
			password: '',
			bio: '',
			preferredLanguage: '',
			timezone: ''
		};

		profilePictureUrl = '';
		if (fileInput) {
			fileInput.value = '';
		}
	}
</script>

<section class="flex min-h-screen items-center justify-center bg-gray-900 px-4">
	<div
		class="w-full max-w-lg space-y-8 rounded-lg border border-gray-700 bg-gray-800 p-8 shadow-2xl"
	>
		<!-- Error Display -->
		{#if error}
			<div class="rounded-md border border-red-500/30 bg-red-900/20 p-4">
				<p class="text-sm text-red-400">{error}</p>
			</div>
		{/if}

		{#if isLogin}
			<div class="space-y-6">
				<h1 class="text-center text-3xl font-bold text-white">Login</h1>
				<form onsubmit={handleLogin} class="space-y-6">
					<div class="space-y-4">
						<div>
							<label for="login-username" class="block text-sm font-medium text-gray-300">
								Username <span class="text-red-400">*</span>
							</label>
							<input
								bind:value={loginFormData.username}
								id="login-username"
								type="text"
								required
								class="w-full rounded-md border border-gray-600 bg-gray-700 px-3 py-2 text-white placeholder-gray-400 shadow-sm focus:border-green-500 focus:outline-none focus:ring-2 focus:ring-green-500/20"
							/>
						</div>
						<div>
							<label for="login-password" class="block text-sm font-medium text-gray-300">
								Password <span class="text-red-400">*</span>
							</label>
							<input
								bind:value={loginFormData.password}
								id="login-password"
								type="password"
								required
								class="w-full rounded-md border border-gray-600 bg-gray-700 px-3 py-2 text-white placeholder-gray-400 shadow-sm focus:border-green-500 focus:outline-none focus:ring-2 focus:ring-green-500/20"
							/>
						</div>
					</div>
					<button
						type="submit"
						disabled={isLoading}
						class="w-full rounded-md bg-green-600 px-4 py-2 text-white shadow-sm transition-colors hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-green-500/20 disabled:cursor-not-allowed disabled:opacity-50"
					>
						{isLoading ? 'Signing in...' : 'Login'}
					</button>
				</form>
				<div class="flex items-center justify-around gap-2">
					<div class="flex items-center justify-center gap-2">
						<p class="text-sm text-gray-400">Don't have an account?</p>
						<button
							onclick={switchToRegister}
							class="text-sm text-green-400 transition-colors hover:text-green-300 hover:underline"
						>
							Register
						</button>
					</div>
					{@render tryIt()}
				</div>
			</div>
		{:else}
			<div class="space-y-6">
				<h1 class="text-center text-3xl font-bold text-white">Register</h1>

				<div class="flex flex-col items-center">
					<div class="relative">
						<div
							class="group relative h-24 w-24 overflow-hidden rounded-full border-2 border-gray-600 bg-gray-700 shadow-md"
						>
							{#if profilePictureUrl}
								<img
									src={profilePictureUrl}
									alt="Profile Preview"
									class="h-full w-full object-cover"
								/>
							{:else}
								<div class="flex h-full w-full items-center justify-center text-gray-400">
									<User size={40} />
								</div>
							{/if}
						</div>

						<button
							type="button"
							aria-label="Upload profile picture"
							onclick={triggerFileInput}
							class="absolute -right-1 bottom-0 flex h-8 w-8 items-center justify-center rounded-full bg-green-600 text-white shadow-md transition-colors hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-green-500/20"
						>
							<Camera size={20} />
						</button>
					</div>

					<input
						type="file"
						accept="image/*"
						class="hidden"
						bind:this={fileInput}
						onchange={handleFileChange}
					/>
				</div>

				<form onsubmit={handleRegister} class="space-y-4">
					<div class="grid grid-cols-1 gap-4 md:grid-cols-2">
						<div class="col-span-1">
							<label class="block text-sm font-medium text-gray-300" for="register-fullName">
								Full Name <span class="text-red-400">*</span>
							</label>
							<input
								bind:value={registerFormData.fullName}
								id="register-fullName"
								type="text"
								required
								placeholder="Full name"
								class="w-full rounded-md border border-gray-600 bg-gray-700 px-3 py-2 text-white placeholder-gray-400 shadow-sm focus:border-green-500 focus:outline-none focus:ring-2 focus:ring-green-500/20"
							/>
						</div>

						<div class="col-span-1">
							<label class="block text-sm font-medium text-gray-300" for="register-email">
								Email Address <span class="text-red-400">*</span>
							</label>
							<input
								bind:value={registerFormData.email}
								id="register-email"
								type="email"
								required
								placeholder="you@example.com"
								class="w-full rounded-md border border-gray-600 bg-gray-700 px-3 py-2 text-white placeholder-gray-400 shadow-sm focus:border-green-500 focus:outline-none focus:ring-2 focus:ring-green-500/20"
							/>
						</div>

						<div class="col-span-1">
							<label class="block text-sm font-medium text-gray-300" for="register-username">
								Username <span class="text-red-400">*</span>
							</label>
							<input
								bind:value={registerFormData.username}
								id="register-username"
								type="text"
								required
								placeholder="Choose a username"
								class="w-full rounded-md border border-gray-600 bg-gray-700 px-3 py-2 text-white placeholder-gray-400 shadow-sm focus:border-green-500 focus:outline-none focus:ring-2 focus:ring-green-500/20"
							/>
						</div>

						<div class="col-span-1">
							<label class="block text-sm font-medium text-gray-300" for="register-password">
								Password <span class="text-red-400">*</span>
							</label>
							<input
								bind:value={registerFormData.password}
								id="register-password"
								type="password"
								required
								placeholder="Create a password"
								class="w-full rounded-md border border-gray-600 bg-gray-700 px-3 py-2 text-white placeholder-gray-400 shadow-sm focus:border-green-500 focus:outline-none focus:ring-2 focus:ring-green-500/20"
							/>
						</div>

						<div class="col-span-1">
							<label
								class="block text-sm font-medium text-gray-300"
								for="register-preferredLanguage"
							>
								Preferred Language
							</label>
							<input
								bind:value={registerFormData.preferredLanguage}
								id="register-preferredLanguage"
								type="text"
								placeholder="e.g., English"
								class="w-full rounded-md border border-gray-600 bg-gray-700 px-3 py-2 text-white placeholder-gray-400 shadow-sm focus:border-green-500 focus:outline-none focus:ring-2 focus:ring-green-500/20"
							/>
						</div>

						<div class="col-span-1">
							<label class="block text-sm font-medium text-gray-300" for="register-timezone">
								Timezone
							</label>
							<input
								bind:value={registerFormData.timezone}
								id="register-timezone"
								type="text"
								placeholder="e.g., America/New_York"
								class="w-full rounded-md border border-gray-600 bg-gray-700 px-3 py-2 text-white placeholder-gray-400 shadow-sm focus:border-green-500 focus:outline-none focus:ring-2 focus:ring-green-500/20"
							/>
						</div>

						<div class="col-span-1 md:col-span-2">
							<label class="block text-sm font-medium text-gray-300" for="register-bio">
								Bio
							</label>
							<textarea
								bind:value={registerFormData.bio}
								id="register-bio"
								rows="3"
								placeholder="Tell us about yourself"
								class="w-full resize-none rounded-md border border-gray-600 bg-gray-700 px-3 py-2 text-white placeholder-gray-400 shadow-sm focus:border-green-500 focus:outline-none focus:ring-2 focus:ring-green-500/20"
							></textarea>
						</div>
					</div>

					<button
						type="submit"
						disabled={isLoading}
						class="w-full rounded-md bg-green-600 px-4 py-2 text-white shadow-sm transition-colors hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-green-500/20 disabled:cursor-not-allowed disabled:opacity-50"
					>
						{isLoading ? 'Creating account...' : 'Register'}
					</button>
				</form>

				<div class="flex items-center justify-around gap-2">
					<div class="flex items-center justify-center gap-2">
						<p class="text-sm text-gray-400">Already have an account?</p>
						<button
							onclick={switchToLogin}
							class="text-sm text-green-400 transition-colors hover:text-green-300 hover:underline"
						>
							Login
						</button>
					</div>
					{@render tryIt()}
				</div>
			</div>
		{/if}
	</div>
</section>

{#snippet tryIt()}
	<div>
		<a href="/chat" class="text-sm text-green-400 transition-colors hover:text-green-300">
			Try It Now
		</a>
	</div>
{/snippet}
